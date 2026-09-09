---
layout: post
title:  "My Triton From Scratch Part 16: Register Tiles"
date:   2026-09-05 16:00:00 +0000
# categories:
---

In [Part 12: CUDA Tile Layouts]({% post_url 2026-08-22-My_Triton_From_Scratch_Part_12_CUDA_Tile_Layouts %}),
mytriton separated logical output shapes from physical CUDA thread shapes.

In [Part 13: tl.dot Semantics]({% post_url 2026-08-23-My_Triton_From_Scratch_Part_13_Dot_Semantics %}),
matrix multiplication became a first-class typed SSA operation.

In [Part 14: Shared-Memory Tiles]({% post_url 2026-08-29-My_Triton_From_Scratch_Part_14_Shared_Memory_Tiles %}),
canonical matrix loads became cooperative global-to-shared copies.

In [Part 15: CUDA-Core tl.dot]({% post_url 2026-08-30-My_Triton_From_Scratch_Part_15_CUDA_Core_Dot %}),
each CUDA thread used those shared tiles to compute one output element.

That final phrase is the limitation Version 16 removes.

The source language describes a logical result tile such as `C[16, 16]`. It
does not require 256 CUDA threads or say that every thread owns exactly one
element. Version 16 makes that distinction real: a smaller physical thread tile
can cover a larger logical output, with several C values stored in registers
owned by each thread.

## Why one result per thread is too restrictive

Version 15 uses this identity:

```text
logical output elements == CUDA threads == scalar accumulators
```

For a `[4, 8]` tile, 32 output elements map naturally to 32 threads. For a
`[16, 16]` tile, the same policy requires 256 threads and still gives each
thread only one accumulator.

That couples two choices that should be independent:

```text
What C tile does one program instance produce?
How many CUDA threads cooperate on that tile?
```

Multiple results per thread let a kernel use a compact physical thread group
while keeping a larger logical tile. They also increase the amount of arithmetic
performed for each thread's address and control overhead.

Version 16 uses at most 32 threads for a `tl.dot` output and maps the remaining
logical coordinates to per-thread registers.

## Three coordinate spaces

The new execution model has three shapes:

```text
logical output shape:  all C elements produced by the program
thread shape:          physical CUDA coordinates in the block
register shape:        output values owned by one thread
```

For example:

```text
logical_shape = (8, 8)
thread_shape  = (4, 8)
register_shape = (2, 1)
```

There are 64 logical outputs, 32 physical threads, and two output registers per
thread.

For a larger example:

```text
logical_shape = (16, 16)
thread_shape  = (4, 8)
register_shape = (4, 2)
```

Every one of the 32 threads owns eight C elements.

## Describing the mapping

`CudaRegisterTileLayout` makes the relationship explicit:

```python
@dataclass(frozen=True)
class CudaRegisterTileLayout:
    logical_shape: tuple[int, ...]
    thread_shape: tuple[int, ...]
```

Both shapes must be positive rank-2 shapes, and every logical dimension must be
divisible by the corresponding thread dimension.

The register shape is their elementwise quotient:

```python
@property
def register_shape(self) -> tuple[int, ...]:
    return tuple(
        logical_dim // thread_dim
        for logical_dim, thread_dim in zip(
            self.logical_shape,
            self.thread_shape,
            strict=True,
        )
    )
```

The number of C values computed by one thread is:

```python
@property
def registers_per_thread(self) -> int:
    return prod(self.register_shape)
```

## A strided ownership rule

Version 16 chooses this mapping:

```text
logical_coordinate =
    thread_coordinate + register_coordinate * thread_shape
```

The multiplication and addition happen independently on row and column axes.

For an `[8, 8]` output with a `[4, 8]` thread shape, thread `(1, 3)` owns:

```text
register (0, 0) -> logical (1, 3)
register (1, 0) -> logical (5, 3)
```

The row changes by the physical thread-row count, four. The column register
shape is one, so both results remain in column three.

For `[16, 16]` over `[4, 8]`, the same thread owns:

```text
register (0, 0) -> logical (1, 3)
register (0, 1) -> logical (1, 11)
register (1, 0) -> logical (5, 3)
register (1, 1) -> logical (5, 11)
register (2, 0) -> logical (9, 3)
register (2, 1) -> logical (9, 11)
register (3, 0) -> logical (13, 3)
register (3, 1) -> logical (13, 11)
```

The complete mapping is regular, covers every output exactly once, and is easy
to turn into CUDA expressions such as `tile_i + 4` and `tile_j + 8`.

## Choosing the physical thread shape

The layout inference now scans nested SSA regions for `dot` result shapes. If
the observable rank-2 output is produced by a dot, a dedicated policy chooses
its physical thread shape.

The current limit is deliberately simple:

```python
CUDA_DOT_MAX_THREADS = 32
```

For each logical dimension, the backend enumerates divisors. It considers every
pair whose product is at most 32 and chooses the maximum according to:

```python
key=lambda shape: (
    prod(shape),
    min(shape),
    shape[1],
)
```

In order, the policy prefers:

1. more threads;
2. a more balanced two-dimensional shape;
3. more columns when the earlier criteria tie.

Some results are:

```text
dot output (4, 8)   -> threads (4, 8), registers (1, 1)
dot output (8, 8)   -> threads (4, 8), registers (2, 1)
dot output (16, 16) -> threads (4, 8), registers (4, 2)
dot output (16, 32) -> threads (4, 8), registers (4, 4)
dot output (3, 17)  -> threads (1, 17), registers (3, 1)
dot output (1, 64)  -> threads (1, 32), registers (1, 2)
```

This is a deterministic correctness policy, not an autotuner. The number 32
means that the current dot path uses at most one warp; it does not claim that
one warp is optimal for every tile.

Ordinary non-dot elementwise kernels keep their earlier one-thread-per-result
layout. Version 16 changes the execution model only where the backend has been
extended to understand register tiles.

## Representing a register tile during code generation

Previously, every SSA value lowered to one of a few scalar-like references:

```text
str
CudaPtrRef
CudaArangeRef
```

Version 16 adds:

```python
@dataclass(frozen=True)
class CudaRegisterTileRef:
    base: str
    layout: CudaRegisterTileLayout
    broadcast_axes: tuple[int, ...] = ()
```

For a result `%29` with register shape `(2, 1)`, it names two CUDA locals:

```text
v29_0_0
v29_1_0
```

For `(4, 2)`, it names eight:

```text
v29_0_0  v29_0_1
v29_1_0  v29_1_1
v29_2_0  v29_2_1
v29_3_0  v29_3_1
```

When the register shape is `(1, 1)`, the old scalar name `v29` is preserved.
That keeps existing small-tile CUDA source stable and avoids making every
operation pay for a collection abstraction.

Code generation can now ask for a value at a register coordinate:

```python
value.element((register_row, register_column))
```

It validates both the coordinate and the expected layout, preventing an SSA
operation from accidentally combining values distributed in incompatible ways.

## Computing several dot results

The shared-memory data movement from Version 14 does not need to change. The
same `A[BM, BK]` and `B[BK, BN]` tiles are visible to all threads after the
barrier.

The FMA lowering now declares one accumulator for every register coordinate:

```c++
float v29_0_0 = 0.0f;
float v29_1_0 = 0.0f;

for (int dot_k_29 = 0; dot_k_29 < BK; ++dot_k_29) {
    v29_0_0 += shared_A[(tile_i) * BK + dot_k_29]
              * shared_B[dot_k_29 * BN + tile_j];

    v29_1_0 += shared_A[(tile_i + 4) * BK + dot_k_29]
              * shared_B[dot_k_29 * BN + tile_j];
}
```

For a two-dimensional `(4, 2)` register tile, the loop contains eight updates.
Each update uses the logical row and column derived from its register
coordinate.

The physical thread still cooperates in loading all of shared A and B. It now
performs more arithmetic before the second barrier.

## Broadcasts cannot be treated as full register tiles

Matmul output addressing begins with two broadcast-shaped values:

```python
offsets_m = block_m + tl.arange(0, BM)[:, None]  # [BM, 1]
offsets_n = block_n + tl.arange(0, BN)[None, :]  # [1, BN]
```

If `C[16, 16]` uses register shape `(4, 2)`, a naive implementation might
materialize eight copies of both values per thread. But `offsets_m` does not
vary along columns, and `offsets_n` does not vary along rows.

`CudaRegisterTileRef.broadcast_axes` records those invariants.

For a `[BM, 1]` value, column axis 1 is broadcast:

```text
logical register shape = (4, 2)
storage shape          = (4, 1)
```

For `[1, BN]`, row axis 0 is broadcast:

```text
logical register shape = (4, 2)
storage shape          = (1, 2)
```

Looking up a broadcast coordinate redirects it to zero on that axis. Thus
`offsets_m` stores four row values, `offsets_n` stores two column values, and a
later operation can still request either at any of the eight logical register
coordinates.

This is both a correctness rule and a compact representation of rank-2
broadcast semantics.

## Expanding aranges into register coordinates

With one result per thread, the CUDA backend could lower:

```python
tl.arange(0, BM)[:, None]
```

to the scalar `tile_i`. That is no longer sufficient when the same thread owns
rows `tile_i`, `tile_i + 4`, `tile_i + 8`, and `tile_i + 12`.

For register-tile kernels, `emit_expand_dims` creates the non-broadcast storage
coordinates explicitly:

```c++
int v3_0_0 = tile_i;
int v3_1_0 = tile_i + 4;
int v3_2_0 = tile_i + 8;
int v3_3_0 = tile_i + 12;
```

The corresponding column arange might produce:

```c++
int v22_0_0 = tile_j;
int v22_0_1 = tile_j + 8;
```

These are the raw logical coordinates from which pointers and masks are built.

## Register-wise binary operations

Once an SSA value may represent several CUDA locals, ordinary arithmetic must
select matching elements.

For a full output tile, a binary add is conceptually:

```python
for register_coordinate in register_layout:
    result[register_coordinate] = (
        lhs[register_coordinate] + rhs[register_coordinate]
    )
```

The lowering accepts either a register tile or a scalar operand. A scalar is
broadcast to every register automatically. A broadcast register tile maps the
requested coordinate through its `broadcast_axes`.

This supports all of the combinations needed by matmul:

```text
scalar + row tile
scalar + column tile
row tile + column tile -> full output tile
accumulator tile + dot result tile
row mask & column mask -> full mask tile
```

For example, adding `[BM, 1]` row coordinates to `[1, BN]` column coordinates
materializes the complete register tile because the result varies along both
axes.

## Pointer tiles

The output pointer expression is also register-valued:

```python
output_pointers = out + offsets_m * N + offsets_n
```

`emit_addptr` now has two paths. The old path produces one `CudaPtrRef`. The new
path walks register coordinates and creates a `CudaRegisterTileRef` whose
elements are CUDA pointer locals:

```c++
float* v32_0_0 = out + offset_0_0;
float* v32_0_1 = out + offset_0_1;
...
```

Pointer lookup then converts a selected register element to:

```python
CudaPtrRef(base=selected_pointer_name, index="0")
```

The same layout compatibility check used for arithmetic prevents a pointer tile
from being indexed with unrelated register coordinates.

## Masks and stores

The output mask combines broadcast row and column comparisons:

```python
output_mask = (offsets_m < M) & (offsets_n < N)
```

Register-wise comparison and Boolean `and` produce one mask value for every C
coordinate owned by the thread.

`emit_store` finds a register-tile operand among the pointer, value, and mask.
It then walks the complete register layout and selects corresponding elements
from all three:

```python
for register_coordinate in register_layout:
    ptr = register_pointer_operand(...)
    value = register_expression_operand(...)
    mask = register_expression_operand(...)
```

The generated CUDA contains one independently masked store per output:

```c++
if (mask_0_0) {
    pointer_0_0[0] = accumulator_0_0;
}
if (mask_1_0) {
    pointer_1_0[0] = accumulator_1_0;
}
```

That is necessary on edge tiles. One thread may own both an in-bounds and an
out-of-bounds logical row or column, so a single thread-level mask is no longer
enough.

## Register-valued loop-carried accumulators

The runtime K loop adds one more complication. Its carried accumulator has
logical type `block<BMxBN x f32>`, but now that one SSA value corresponds to
several CUDA variables.

Before the loop, Version 16 initializes every owned register from the carried
input:

```c++
float acc_0_0 = 0.0f;
float acc_1_0 = 0.0f;
```

Inside the loop, `acc + tl.dot(...)` produces another register tile:

```c++
float next_0_0 = acc_0_0 + dot_0_0;
float next_1_0 = acc_1_0 + dot_1_0;
```

The structured `yield` is lowered coordinate by coordinate:

```c++
acc_0_0 = next_0_0;
acc_1_0 = next_1_0;
```

After the loop, the SSA result maps back to the same register-tile reference
and the final store consumes all of its elements.

This preserves the semantics established in Part 10:

```text
one SSA carried value
```

while allowing its CUDA representation to be:

```text
several physical registers per thread
```

SSA value count and machine register count are no longer assumed to be equal.

## A complete `(8, 8)` example

Take:

```text
BM = 8
BN = 8
BK = 8
```

Layout inference chooses:

```text
output tile    = (8, 8)
thread tile    = (4, 8)
register tile  = (2, 1)
threads        = 32
outputs/thread = 2
```

The CUDA prologue remains:

```c++
int tile_i = threadIdx.x / 8;
int tile_j = threadIdx.x % 8;
```

but the two logical output coordinates are:

```text
(tile_i,     tile_j)
(tile_i + 4, tile_j)
```

The dot loop accumulates both. Output pointer arithmetic builds both addresses,
the mask checks both coordinates, and the store writes each value separately.

The same source kernel did not change. Only the backend ownership mapping did.

## A two-dimensional register tile

For `C[16, 16]`, layout inference still chooses 32 threads:

```text
thread shape   = (4, 8)
register shape = (4, 2)
```

Each thread owns four row positions and two column positions. The Cartesian
product yields eight accumulators.

This case is important because it proves the implementation is not merely a
special “two rows per thread” trick. Aranges, broadcasting, arithmetic,
pointers, masks, loop-carried values, dot computation, and stores all work on a
genuine two-dimensional register domain.

## Tests cover the mapping as a contract

Layout unit tests verify:

- positive rank-2 shapes;
- divisibility between logical and thread dimensions;
- register shapes and registers per thread;
- exact logical coordinates;
- invalid thread and register coordinates;
- the one-warp thread-shape policy;
- unchanged layouts for non-dot kernels.

Code-generation unit tests cover `CudaRegisterTileRef` naming and broadcast
storage, register-wise binary operations, pointer addition, Boolean masks,
masked stores, expanded aranges, and incompatible-layout errors.

Dot tests assert that a smaller thread tile emits several accumulator updates.
Loop tests verify that every register is initialized, updated by the yielded
value, and available after the runtime loop.

AST-to-CUDA tests compare exact source for a multi-result kernel. GPU execution
tests compare against matrix multiplication for:

- one K tile;
- several K tiles;
- partial M, N, and K edges;
- an `(8, 8)` output with two results per thread;
- a `(16, 16)` output with a `(4, 2)` register tile.

The two-dimensional execution test is the end-to-end proof that every stage
agrees on the same ownership mapping.

## What Version 16 does not optimize

Register tiles remove an architectural restriction, but the policy remains
intentionally naive.

Version 16 does not choose layouts using GPU occupancy, register pressure,
memory throughput, or instruction scheduling. It simply searches divisors and
uses at most one warp.

The shared-memory layout is still flat row-major with no padding or swizzling.
Cooperative global loads are scalar. There is no double buffering or overlap
between copies and computation. Inputs and accumulators remain `f32`, and the
dot loop uses ordinary CUDA cores rather than tensor cores.

Those are later performance and numeric-model problems. Version 16 establishes
the representation they need: explicit ownership of multiple logical values by
one physical thread.

## What changed conceptually

Before Version 16, CUDA lowering treated an SSA block as one scalar value in
each thread:

```text
block<BMxBN> + thread coordinate
          -> one CUDA local
```

After Version 16, a dot output has an explicit distribution:

```text
logical C tile
      |
      +-- physical thread coordinate
      |
      +-- per-thread register coordinate
                    |
                    v
           exact logical C coordinate
```

That mapping propagates through the entire output-side dataflow:

```text
expanded aranges
    -> broadcast row/column register tiles
    -> arithmetic and pointer register tiles
    -> per-register masks
    -> multiple dot accumulators
    -> register-valued loop carries
    -> masked stores
```

The source still contains one `tl.dot` and one logical accumulator. The CUDA
backend now has enough structure to realize them as several registers per
thread.

All code for this milestone is available at
[https://github.com/pbelevich/mytriton/tree/ver16](https://github.com/pbelevich/mytriton/tree/ver16).

Next: [Part 17: PyTorch Tensor Interoperability]({% post_url 2026-09-06-My_Triton_From_Scratch_Part_17_PyTorch_Tensor_Interoperability %}).
