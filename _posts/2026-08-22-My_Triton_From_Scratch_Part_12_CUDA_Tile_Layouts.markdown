---
layout: post
title:  "My Triton From Scratch Part 12: CUDA Tile Layouts"
date:   2026-08-22 16:00:00 +0000
# categories:
---

In [Part 1: Symbolic Tracing]({% post_url 2026-06-22-My_Triton_From_Scratch_Part_1_Symbolic_Tracing %}),
mytriton learned how to build a symbolic expression tree from a kernel.

In [Part 2: Typed SSA]({% post_url 2026-06-23-My_Triton_From_Scratch_Part_2_Typed_SSA %}),
that tree gained explicit types and SSA values.

In [Part 3: CUDA Lowering]({% post_url 2026-06-24-My_Triton_From_Scratch_Part_3_CUDA_Lowering %}),
SSA became CUDA C++ and an executable kernel.

In [Part 4: Elementwise Ops]({% post_url 2026-06-25-My_Triton_From_Scratch_Part_4_Elementwise_Ops %}),
the language grew a useful elementwise vocabulary.

In [Part 5: Verification]({% post_url 2026-06-26-My_Triton_From_Scratch_Part_5_Verification %}),
the compiler began checking and optimizing its SSA contract.

In [Part 6: Reductions]({% post_url 2026-06-27-My_Triton_From_Scratch_Part_6_Reductions %}),
CUDA threads learned to cooperate through shared-memory reduction scratch space.

In [Part 7: Minimal MLIR]({% post_url 2026-06-28-My_Triton_From_Scratch_Part_7_Minimal_MLIR %}),
the same optimized SSA gained a second lowering path.

In [Part 8: Rank-2 Tiles]({% post_url 2026-06-29-My_Triton_From_Scratch_Part_8_Rank_2_Tiles %}),
logical blocks gained rows, columns, and broadcasting.

In [Part 9: AST Frontend]({% post_url 2026-08-01-My_Triton_From_Scratch_Part_9_AST_Frontend %}),
mytriton took ownership of Python kernel syntax.

In [Part 10: Runtime For Loops]({% post_url 2026-08-08-My_Triton_From_Scratch_Part_10_Runtime_For_Loops %}),
runtime iteration became a structured SSA region with carried values.

In [Part 11: Block Factory Functions]({% post_url 2026-08-15-My_Triton_From_Scratch_Part_11_Block_Factory_Functions %}),
kernels gained `tl.empty`, `tl.full`, and `tl.zeros` logical block constructors.

Those pieces are enough for a naive matrix multiplication, but not yet for a
good one.

Version 12 takes the next architectural step: logical block shapes stop doubling
as an implicit description of CUDA threads.

## One shape was doing too many jobs

The rank-2 CUDA backend previously inferred one `block_shape` from every
block-typed SSA result. For a `(4, 8)` output tile it launched 32 threads and
decoded the linear thread ID as:

```c++
int tile_i = threadIdx.x / 8;
int tile_j = threadIdx.x % 8;
```

That works for the current elementwise policy:

```text
logical output tile: 4 x 8
CUDA thread grid:    4 x 8
ownership:           one output element per thread
```

But “shape” is answering several different questions here:

1. What is the logical shape and dtype of an SSA value?
2. What output domain does one program instance write?
3. How many CUDA threads execute the program instance?
4. Which logical element does each thread own?
5. How do all threads cooperate on a tile that has a different shape?

Those answers happen to coincide for a simple elementwise tile. They do not
coincide for shared-memory matmul.

Suppose one program instance computes:

```text
C tile: [BM, BN]
```

while each `K` iteration consumes:

```text
A tile: [BM, BK]
B tile: [BK, BN]
```

There is no reason for `BM * BN`, `BM * BK`, `BK * BN`, and the number of CUDA
threads to be equal. A useful matmul may compute a `64 x 64` output tile with
256 threads. Those threads must collectively load `A` and `B`, then each thread
must own several accumulator elements.

Before implementing shared-memory loads or `tl.dot`, the compiler needs words
for these different mappings.

## Logical type versus execution layout

`BlockType` continues to describe a value visible to the language and SSA:

```python
BlockType(shape=(BM, BK), element=F32)
```

It says that the value is logically a `BM x BK` tile of floating-point elements.
It deliberately does not say which CUDA thread owns an element or where the
tile is stored.

Version 12 introduces three CUDA-specific layout objects:

```text
CudaKernelLayout
    output_tile_shape
    thread_shape

CudaTileLayout
    logical_shape
    thread_axes

CudaCooperativeTileLayout
    logical_shape
    threads_per_block
    order
```

They answer related but distinct questions:

- `CudaKernelLayout` describes the program instance as a whole;
- `CudaTileLayout` projects a per-thread logical value onto thread coordinates;
- `CudaCooperativeTileLayout` distributes an arbitrary tile over all threads.

## Kernel layout

The top-level object separates the output domain from physical thread
organization:

```python
@dataclass(frozen=True)
class CudaKernelLayout:
    output_tile_shape: tuple[int, ...]
    thread_shape: tuple[int, ...]
```

For the current rank-2 matmul:

```python
layout = CudaKernelLayout(
    output_tile_shape=(4, 8),
    thread_shape=(4, 8),
)
```

The layout has convenient derived properties:

```python
layout.rank               # 2
layout.is_rank2           # True
layout.threads_per_block  # 32
```

The model can also state a future ownership policy that uses fewer threads than
output elements:

```python
layout = CudaKernelLayout(
    output_tile_shape=(64, 64),
    thread_shape=(8, 32),
)
```

This says that a `64 x 64` output tile is assigned to 256 threads. Version 12
can represent and validate that separation, but automatic inference and CUDA
emission do not yet implement multi-element output ownership. For current
elementwise rank-2 kernels, inferred `thread_shape` remains equal to
`output_tile_shape`.

Both shapes must have rank one or two, have positive exact-integer dimensions,
and have the same rank. Their physical thread product may not exceed CUDA's
1024-thread block limit.

## Output layout should come from observable output

Before Version 12, shape inference collected every block-typed SSA result:

```python
if op.result is not None and isinstance(op.result.ty, BlockType):
    shapes.append(op.result.ty.shape)
```

That made internal implementation details influence kernel launch geometry. A
temporary `zeros((2, 3), ...)` could conflict with a `(4, 8)` store even though
the temporary was not the output domain. Future `A[BM, BK]` and `B[BK, BN]`
tiles would make the problem unavoidable.

Version 12 roots output-layout inference at observable memory writes:

```python
def store_block_shapes(ssa_ops: list[SSAItem]) -> list[tuple[int, ...]]:
    shapes = []

    for op in ssa_ops:
        if isinstance(op, SSAForRange):
            shapes.extend(store_block_shapes(op.body))
            continue

        if op.opcode != "store":
            continue

        for operand in op.operands:
            if isinstance(operand, SSAValue) and isinstance(operand.ty, BlockType):
                shapes.append(operand.ty.shape)

    return shapes
```

The pointer, value, and mask of a store all participate because together they
describe its observable domain. Compatible shapes are broadcast to one output
tile.

For example:

```text
store pointer: block<4x8 x ptr<f32>>
store value:   block<4x8 x f32>
store mask:    block<4x8 x bool>
```

infers:

```text
output_tile_shape = (4, 8)
```

An unrelated internal value no longer changes the launch layout:

```text
%0 = zeros ... : block<2x3 x f32>
store %ptr_4x8, %value_4x8, %mask_4x8
```

still has output tile `(4, 8)`.

Store discovery recurses into runtime loop bodies. A store remains observable
even when it is nested in structured control flow.

The current policy rejects mixed rank-1 and rank-2 store domains. Multiple
rank-1 stores must use one width; rank-2 store shapes must broadcast to one
rank-2 shape. Kernels with no block-shaped store operands fall back to the
scalar output tile `(1,)`.

## Output shape is not always thread shape

Store-rooted inference fixes internal tiles, but reductions reveal another
distinction.

Consider a block reduction that stores one scalar:

```python
offsets = tl.arange(0, BLOCK)
values = tl.load(x + offsets)
total = tl.sum(values)
tl.store(out, total)
```

The observable output is scalar:

```text
output_tile_shape = (1,)
```

The reduction still needs `BLOCK` cooperative CUDA threads. Launching one thread
would make the existing shared-memory tree reduction wrong.

Version 12 therefore separately scans reduction inputs:

```python
def reduction_block_shapes(ssa_ops):
    ...
    if op.opcode in {"sum", "max", "min"}:
        shapes.append(op.operands[0].ty.shape)
```

For a four-element reduction, layout inference produces:

```text
output_tile_shape = (1,)
thread_shape      = (4,)
threads_per_block = 4
```

This is the first automatically inferred layout in mytriton where logical
output shape and CUDA thread shape differ.

Reduction inputs must still be rank one, and all reductions in a kernel must
agree on one width. For a rank-2 output, that width must equal the number of
output elements under the current one-thread-per-element lowering.

The complete inference pipeline is now:

```python
def cuda_kernel_layout(ssa_ops):
    output_tile_shape = _infer_cuda_kernel_tile_shape(ssa_ops)
    thread_shape = _infer_cuda_thread_shape(
        output_tile_shape,
        reduction_block_shapes(ssa_ops),
    )
    return CudaKernelLayout(output_tile_shape, thread_shape)
```

## Projecting logical values onto threads

The output tile is not the only block value in a rank-2 kernel. Broadcasting
creates shapes such as:

```text
full tile:       (BM, BN)
row coordinate:  (BM, 1)
column coordinate: (1, BN)
```

For the current one-element-per-thread mapping, a `CudaTileLayout` records which
logical axes are supplied by which thread axes:

```python
@dataclass(frozen=True)
class CudaTileLayout:
    logical_shape: tuple[int, ...]
    thread_axes: tuple[int | None, ...]
```

For a `(4, 8)` thread shape:

```text
logical (4, 8) -> thread_axes (0, 1)
logical (4, 1) -> thread_axes (0, none)
logical (1, 8) -> thread_axes (none, 1)
```

`None` means that the logical dimension is broadcast and does not vary with a
CUDA thread coordinate.

The layout projection validates each dimension:

```python
if logical_dim == thread_dim:
    thread_axes.append(axis)
elif logical_dim == 1:
    thread_axes.append(None)
else:
    raise ValueError(...)
```

It rejects shapes such as `(4, 3)` for a `(4, 8)` thread organization because
there is no direct per-thread projection for the second dimension.

## Explicit broadcast axes

Dimension size alone is not always enough to recover intent. Imagine a kernel
whose thread shape is `(1, 8)`. A logical `(1, 1)` value has size one on both
axes, but one singleton corresponds to a real, degenerate thread axis and the
other was introduced by broadcasting.

`tile_layout` therefore accepts explicit `broadcast_axes`:

```python
layout.tile_layout(
    logical_shape=(1, 1),
    broadcast_axes=(1,),
)
```

and returns:

```python
CudaTileLayout(
    logical_shape=(1, 1),
    thread_axes=(0, None),
)
```

Axis zero remains mapped even though its extent is one. Axis one is explicitly
broadcast.

This matters in CUDA lowering of `expand_dims`. The operation itself records
which axis was inserted, so code generation passes that fact to the layout:

```python
tile_layout = self.layout.tile_layout(
    result_shape,
    broadcast_axes=(axis,),
)
```

The mapped non-broadcast axis selects `tile_i` or `tile_j`. The code generator
no longer contains special cases hard-coded specifically for `(rows, 1)` and
`(1, cols)`.

## CUDA thread coordinates

`SSACUDACodegen` now owns a `CudaKernelLayout` rather than a raw tuple:

```python
self.layout = cuda_kernel_layout(ssa_ops)
```

For rank one, the only thread coordinate is:

```text
threadIdx.x
```

For rank two, the existing prologue is derived from `thread_shape`:

```c++
int tile_i = threadIdx.x / cols;
int tile_j = threadIdx.x % cols;
```

and `thread_coordinate(axis)` maps axis zero to `tile_i` and axis one to
`tile_j`.

This refactoring intentionally preserves generated CUDA for existing kernels.
Version 12 changes how the backend reasons about mapping, not the elementwise
ownership policy itself.

## Cooperative tile layouts

A direct `CudaTileLayout` only works when each logical dimension either matches
one thread dimension or broadcasts. Matmul operands do not satisfy that rule.

For example, with a `(4, 8)` output/thread layout and `BK = 16`:

```text
A tile = (4, 16)   # cannot project directly onto (4, 8)
B tile = (16, 8)   # cannot project directly onto (4, 8)
```

These tiles must be traversed cooperatively by the whole CUDA block. That is
the role of `CudaCooperativeTileLayout`:

```python
@dataclass(frozen=True)
class CudaCooperativeTileLayout:
    logical_shape: tuple[int, ...]
    threads_per_block: int
    order: tuple[int, ...]
```

Thread `t` owns the following linear indices:

```text
t
t + threads_per_block
t + 2 * threads_per_block
...
```

or in code:

```python
def linear_index(self, thread_index: int, iteration: int) -> int:
    return thread_index + iteration * self.threads_per_block
```

The number of iterations per thread is ceiling division:

```python
iterations_per_thread = ceil(tile_size / threads_per_block)
```

If the tile size is not divisible by the number of threads, `contains` identifies
inactive tail lanes.

The default rank-2 order is `(1, 0)`, which means axis one changes fastest. It
is the familiar row-major coordinate conversion:

```text
row = linear / columns
column = linear % columns
```

The generic implementation applies the axes in `order`:

```python
for axis in self.order:
    coordinates[axis] = remaining % self.logical_shape[axis]
    remaining //= self.logical_shape[axis]
```

## Cooperative `A` tile example

Take 32 threads and an `A` tile of shape `(4, 16)`:

```python
a_layout = kernel_layout.cooperative_tile_layout((4, 16))
```

The tile has 64 elements, so every thread handles two:

```text
size                  = 64
threads_per_block     = 32
iterations_per_thread = 2
```

Thread zero visits linear indices 0 and 32:

```text
linear 0  -> coordinate (0, 0)
linear 32 -> coordinate (2, 0)
```

Thread 31 visits 31 and 63:

```text
linear 31 -> coordinate (1, 15)
linear 63 -> coordinate (3, 15)
```

Across the block, every element of `A[4, 16]` is covered exactly once.

## Cooperative `B` tile example

For `B[16, 8]`, the same 32 threads cover 128 elements, so every thread handles
four:

```python
b_layout = kernel_layout.cooperative_tile_layout((16, 8))
```

Thread zero visits:

```text
linear 0  -> (0, 0)
linear 32 -> (4, 0)
linear 64 -> (8, 0)
linear 96 -> (12, 0)
```

Again, the tile shape is independent of both output shape and thread shape.
That is the capability the old single `block_shape` abstraction could not
express.

## Validation is part of the model

Layout objects reject invalid states when constructed:

- logical shapes must have rank one or two and positive dimensions;
- thread counts must be integers from 1 through 1024;
- projected thread axes must be in range and unique;
- logical and thread-axis tuples must have equal rank;
- cooperative `order` must be a permutation of all logical axes;
- thread and iteration indices must be in range;
- coordinate conversion rejects linear indices outside the tile.

These are not merely defensive checks. A wrong layout would generate valid C++
that reads or writes the wrong memory. Making mapping objects immutable and
validated gives later lowering stages a contract they can rely on.

## Tests as executable layout documentation

Version 12 adds focused tests for all three layers:

- kernel rank, thread count, and shape validation;
- store-rooted inference that ignores unrelated internal blocks;
- reduction-aware thread shape for scalar output;
- rejection of mixed rank-1/rank-2 store domains;
- full, row-broadcast, and column-broadcast projections;
- preservation of a degenerate non-broadcast axis;
- rejection of unmappable logical tiles;
- explicit separation of a `64 x 64` output from `8 x 32` threads;
- cooperative traversal of `A[4, 16]` and `B[16, 8]`;
- inactive tail lanes for a tile smaller than the thread block;
- invalid axis, order, thread, iteration, and 1024-thread-limit cases.

Existing kernel tests still check exact CUDA source. That confirms the layout
refactoring has not accidentally changed current address or coordinate
semantics.

## What Version 12 does not do

The cooperative mapping is a model and a tested indexing primitive in this
milestone. CUDA code generation does not yet use it to emit cooperative global
loads.

There is still no:

- `tl.dot` operation in the language or SSA;
- shared-memory tile allocation for matmul operands;
- cooperative global-to-shared copy generated from a layout;
- synchronization around those copies;
- per-thread accumulator fragment layout;
- shared-memory bank-conflict policy;
- double buffering or asynchronous copy;
- tensor-core fragment and instruction lowering.

`tl.empty`, `tl.full`, and `tl.zeros` continue to create logical distributed
values. They do not become shared-memory arrays because a cooperative layout
exists.

The current naive matmul still loads directly from global memory on each `K`
iteration and performs scalar multiplication in every output thread.

## What changed conceptually

Before Version 12, the CUDA backend inferred one tuple and treated it as both
the logical tile and the physical block:

```text
all SSA block results
        |
        v
one block_shape
        |
        +-- output domain
        +-- thread count
        +-- thread coordinates
```

Now those responsibilities are explicit:

```text
store operands --------------------> output_tile_shape
reduction inputs ------------------> thread_shape
                                      |
                                      v
                               CudaKernelLayout
                                  /         \
                                 v           v
                    projected per-thread   cooperative tile
                         layout              layout
```

The automatically generated elementwise code mostly behaves as before. The
architectural change is that “logical tile” and “CUDA threads” are no longer
synonyms.

That gives the next matmul stages somewhere to put their semantics. A
cooperative loader can ask a layout which `A` or `B` elements each thread moves.
A shared-memory lowering can use those coordinates to choose addresses. A
future `tl.dot` lowering can separately define accumulator ownership and,
eventually, map fragments to tensor-core instructions.

Version 12 does not implement those stages, but it removes the assumption that
would have made them impossible to express cleanly.

All code for this milestone is available at
[https://github.com/pbelevich/mytriton/tree/ver12](https://github.com/pbelevich/mytriton/tree/ver12).

The next milestone will start using these layouts for the data movement needed
by a shared-memory `tl.dot` implementation. Tensor cores come later, after the
ordinary shared-memory path makes the operation's shape, dtype, ownership, and
synchronization semantics explicit.
