---
layout: post
title:  "My Triton From Scratch Part 14: Shared-Memory Tiles"
date:   2026-08-29 16:00:00 +0000
# categories:
---

In [Part 8: Rank-2 Tiles]({% post_url 2026-06-29-My_Triton_From_Scratch_Part_8_Rank_2_Tiles %}),
mytriton learned to represent matrices as logical blocks.

In [Part 9: AST Frontend]({% post_url 2026-08-01-My_Triton_From_Scratch_Part_9_AST_Frontend %}),
the compiler began interpreting Python kernel syntax itself.

In [Part 10: Runtime For Loops]({% post_url 2026-08-08-My_Triton_From_Scratch_Part_10_Runtime_For_Loops %}),
runtime `range` loops became structured SSA regions.

In [Part 11: Block Factory Functions]({% post_url 2026-08-15-My_Triton_From_Scratch_Part_11_Block_Factory_Functions %}),
`tl.zeros` gave matmul an explicit accumulator.

In [Part 12: CUDA Tile Layouts]({% post_url 2026-08-22-My_Triton_From_Scratch_Part_12_CUDA_Tile_Layouts %}),
logical tiles were separated from the CUDA threads that process them.

In [Part 13: tl.dot Semantics]({% post_url 2026-08-23-My_Triton_From_Scratch_Part_13_Dot_Semantics %}),
matrix multiplication became a typed and verified SSA operation.

The compiler now understands that:

```text
dot(block<BMxBK x f32>, block<BKxBN x f32>)
    -> block<BMxBN x f32>
```

It still does not know how the two input tiles reach the fast memory shared by
a CUDA block. Version 14 implements that data movement.

This milestone deliberately stops before multiplication. It recognizes a
canonical pair of masked matrix loads, reconstructs their two-dimensional
meaning from SSA, cooperatively copies both tiles into shared memory, and
synchronizes the block. The resulting CUDA fragment ends with an explicit
diagnostic because the FMA loop belongs to Version 15.

## Why stage through shared memory

Consider one output tile `C[BM, BN]`. Computing it requires:

```text
A tile: [BM, BK]
B tile: [BK, BN]
```

Every `A[m, k]` value participates in several output columns, and every
`B[k, n]` value participates in several output rows. Reading both values from
global memory for every multiply wastes that reuse.

Shared-memory matmul changes the data path:

```text
global A ---- cooperative load ----> shared A --\
                                                   future dot
global B ---- cooperative load ----> shared B --/
```

Threads first cooperate to move each input element once per tile. After a
barrier, the computation can reuse those values many times from shared memory.

The word *staging* refers to this movement and temporary placement. It does not
mean multiplication:

```text
matching  = understand which global tile an SSA value describes
staging   = emit global-to-shared copies for that tile
compute   = consume shared A and shared B to produce C
```

Version 14 implements the first two boxes.

## Shared buffers are physical storage

`tl.zeros` creates a logical block value. It does not allocate CUDA shared
memory. Version 14 therefore introduces a backend-only representation:

```python
@dataclass(frozen=True)
class CudaSharedBuffer:
    name: str
    logical_shape: tuple[int, ...]
    element_ty: ScalarType
```

The initial implementation accepts positive rank-2 tiles. It derives the usual
properties:

```python
@property
def rows(self) -> int:
    return self.logical_shape[0]

@property
def columns(self) -> int:
    return self.logical_shape[1]

@property
def size(self) -> int:
    return self.rows * self.columns

@property
def nbytes(self) -> int:
    return self.size * cuda_scalar_nbytes(self.element_ty)
```

CUDA stores the tile as one flat row-major array. A logical coordinate becomes:

```python
def element(self, row: str, column: str) -> str:
    return f"{self.name}[({row}) * {self.columns} + ({column})]"
```

For `A[4, 16]`, the declaration is:

```c++
__shared__ float dot_lhs_29[64];
```

For `B[16, 8]`:

```c++
__shared__ float dot_rhs_29[128];
```

This is a physical CUDA allocation, unlike the logical values in the source
language.

## A global tile description

To generate the copy, the backend needs more than a pointer. It needs to know
where the tile starts, how matrix rows are laid out, and where the valid matrix
ends.

`CudaGlobalTile` records the resolved CUDA expressions:

```python
@dataclass(frozen=True)
class CudaGlobalTile:
    base: str
    row_offset: str
    column_offset: str
    row_stride: str
    row_bound: str
    column_bound: str
    other: str = "0.0f"
```

For an `A` tile in matmul, those fields might be:

```text
base          = a
row_offset    = blockIdx.x * BM
column_offset = k_base
row_stride    = K
row_bound     = M
column_bound  = K
other         = 0.0f
```

The matching phase first stores equivalent SSA operands in a
`CudaGlobalTilePlan`. Code generation resolves those operands to CUDA strings
only when their definitions are available in the current scope.

Keeping a plan separate from a resolved tile is useful inside runtime loops:
`k_base` may be an SSA region argument rather than a top-level parameter.

## The canonical source pattern

Version 14 does not attempt arbitrary pointer analysis. It recognizes one
clear, Triton-like matrix tile pattern:

```python
offsets_m = tl.program_id(0) * BM + tl.arange(0, BM)[:, None]
offsets_n = tl.program_id(1) * BN + tl.arange(0, BN)[None, :]
offsets_k = tl.arange(0, BK)

a_rows = offsets_m
a_columns = k_base + offsets_k[None, :]
a_values = tl.load(
    a + a_rows * K + a_columns,
    mask=(a_rows < M) & (a_columns < K),
    other=0.0,
)

b_rows = k_base + offsets_k[:, None]
b_columns = offsets_n
b_values = tl.load(
    b + b_rows * N + b_columns,
    mask=(b_rows < K) & (b_columns < N),
    other=0.0,
)

result = tl.dot(a_values, b_values)
```

Each pointer has the row-major form:

```text
base + rows * row_stride + columns
```

Each axis has the form:

```text
scalar tile offset + expand_dims(arange(0, size))
```

Each load has a two-dimensional bounds mask and `other=0.0`.

These restrictions are not the full `tl.dot` language contract. They are the
subset for which this CUDA backend knows how to reconstruct a safe cooperative
copy.

## Building a use-def index

The dot operands are SSA values such as `%14` and `%28`. To understand them,
the backend must follow definitions backwards:

```text
%29 = dot %14, %28
          |    |
          |    +-- load -> mask, pointer, other
          |                  |
          |                  +-- addptr -> row stride and column coordinates
          |
          +-- load -> mask, pointer, other
```

`SSADefinitions` walks top-level operations and nested `SSAForRange` bodies. It
maps each result ID to the operation that defines it:

```python
self.ops[result.id] = op
```

It provides two key operations:

```python
definitions.require(value, "load")
definitions.dependency_ids(lhs, rhs)
```

`require` both follows a value and checks the expected opcode.
`dependency_ids` recursively collects the SSA graph needed to produce a set of
operands.

This small use-def index is enough for a narrow structural matcher without
turning CUDA code generation into a second frontend.

## Matching the pointer expression

`CudaDotOperandMatcher.match` starts at a dot operand and requires a `load`:

```text
load(pointer, mask, other)
```

It then peels two pointer additions:

```text
outer addptr: row_pointer + columns
inner addptr: base + rows * row_stride
```

The multiplication may spell its scalar and block operands in either order, so
the matcher accepts both:

```text
rows * stride
stride * rows
```

but one operand must be a block of `i32` coordinates and the other a scalar
`i32` stride.

The base must be a global pointer parameter. Version 14 does not match an
arbitrary derived pointer because doing so would require a more general alias
and address analysis.

## Matching rows and columns

For each coordinate block, `_match_axis_offset` looks for:

```text
add(scalar_offset, expanded_arange)
```

The expanded arange axis determines its matrix role:

```text
rows:    expand_dims(arange(0, rows), axis=1) -> [rows, 1]
columns: expand_dims(arange(0, cols), axis=0) -> [1, cols]
```

For the `A` tile, the recovered origins are:

```text
row_offset    = program_id(0) * BM
column_offset = k_base
```

For `B`:

```text
row_offset    = k_base
column_offset = program_id(1) * BN
```

Notice that the matcher recovers *meaning* from ordinary SSA operations. There
is no special `matrix_load` operation in the frontend IR yet.

## Matching the safety mask

The load mask must have this shape:

```text
(rows < row_bound) & (columns < column_bound)
```

The matcher follows the `and`, identifies which comparison uses the recovered
row coordinates and which uses the columns, and requires scalar `i32` bounds.

For `A`, it reconstructs:

```text
rows < M
columns < K
```

For `B`:

```text
rows < K
columns < N
```

It also requires `other=0.0`. Zero fill is essential for partial `K` tiles:
an out-of-bounds element must contribute nothing to the later dot product.

## Matching is not staging

The distinction is worth making concrete.

Matching this source:

```python
a_values = tl.load(
    a + a_rows * K + a_columns,
    mask=(a_rows < M) & (a_columns < K),
    other=0.0,
)
```

produces a plan equivalent to:

```python
CudaGlobalTilePlan(
    base=a,
    row_offset=block_m,
    column_offset=k_base,
    row_stride=K,
    row_bound=M,
    column_bound=K,
    other=0.0,
)
```

No CUDA statement has been emitted yet. Staging consumes that plan and emits:

```c++
for (int index = threadIdx.x; index < BM * BK; index += threads_per_block) {
    int row = index / BK;
    int column = index % BK;
    int global_row = block_m + row;
    int global_column = k_base + column;
    shared_a[index] =
        global_row < M && global_column < K
            ? a[global_row * K + global_column]
            : 0.0f;
}
```

Matching is analysis. Staging is code generation based on that analysis.

## Deciding which SSA operations are staging-only

The canonical load graph contains many rank-2 operations: expanded aranges,
pointer arithmetic, comparisons, Boolean conjunctions, and the load itself.
The old scalar CUDA lowering would try to emit them as one value per output
thread.

That is wrong for dot operands. `A[BM, BK]` and `B[BK, BN]` are not distributed
like `C[BM, BN]`; they are being traversed by cooperative copy loops.

`CudaDotStagingAnalyzer` therefore computes:

```python
@dataclass(frozen=True)
class CudaDotStagingAnalysis:
    dot_plans: dict[int, CudaDotStagingPlan]
    staging_only_ids: frozenset[int]
```

Operations in `staging_only_ids` are skipped by normal per-thread lowering.
Their meaning is consumed by the matcher and re-emitted as the cooperative
load.

This is a small form of lowering selection:

```text
canonical dot operand subgraph
          |
          +-- do not scalar-lower private tile operations
          |
          +-- lower their recovered plan as a cooperative copy
```

## Preserving shared dependencies

Not every operation in the load's dependency graph is private to staging.
`offsets_m` and `offsets_n` are usually reused to construct the output pointer
and output mask:

```python
output_pointers = out + offsets_m * N + offsets_n
output_mask = (offsets_m < M) & (offsets_n < N)
```

If the analysis marked those shared definitions as staging-only, the later
store would use undefined CUDA values.

The analyzer therefore subtracts two groups from the private staging graph:

```text
required scalar dependencies
external dependencies used outside staging
```

Only the remaining operations are skipped. This is why staging analysis is not
equivalent to “delete everything reachable from the loads.” It must preserve
the boundary between the specialized dot path and ordinary output lowering.

## Cooperative copy mapping

Part 12 introduced `CudaCooperativeTileLayout`. Version 14 finally uses it.
For a row-major shared tile, every thread starts at its linear thread ID and
advances by the number of threads in the block:

```text
threadIdx.x
threadIdx.x + threads_per_block
threadIdx.x + 2 * threads_per_block
...
```

The emitted CUDA loop is:

```c++
for (int dot_lhs_29_index = threadIdx.x;
     dot_lhs_29_index < 64;
     dot_lhs_29_index += 32) {
    int dot_lhs_29_row = dot_lhs_29_index / 16;
    int dot_lhs_29_column = dot_lhs_29_index % 16;
    // load one logical A element
}
```

For a `[4, 16]` tile and 32 threads, every thread copies two elements. For a
`[16, 8]` tile, every thread copies four.

This mapping works even when the tile size is not divisible by the thread
count: threads whose linear index falls beyond the tile simply execute fewer
iterations.

The first implementation accepts only row-major order `(1, 0)`. Rejecting
other orders is safer than pretending their coordinate calculation is already
implemented.

## Zero-filled boundary tiles

The cooperative loop derives logical and global coordinates separately:

```c++
int row = index / columns;
int column = index % columns;
int global_row = row_offset + row;
int global_column = column_offset + column;
```

It then checks both bounds and always writes shared memory:

```c++
bool in_bounds = global_row < row_bound && global_column < column_bound;
shared[index] = in_bounds
    ? base[global_row * row_stride + global_column]
    : 0.0f;
```

The unconditional shared-memory write is important. All threads must reach the
same barrier, and every shared location consumed by the future computation must
contain a defined value. A partial matrix tile is represented as a full shared
tile padded with zeros.

## Synchronizing the block

The two cooperative loops are followed by:

```c++
__syncthreads();
```

Without this barrier, one thread could begin reading a shared element before
the thread responsible for loading it has written it.

The correct phase ordering is:

```text
all threads load pieces of A
all threads load pieces of B
             |
             v
      __syncthreads()
             |
             v
future shared-memory computation
```

The load masks do not wrap the barrier. Out-of-bounds positions are zero-filled
inside the cooperative loops, so edge blocks still synchronize uniformly.

## Tracking the shared-memory budget

Shared memory is finite. The backend uses a conservative 48 KiB limit and
accounts for every allocation:

```python
required_bytes = self.shared_memory_bytes + additional_bytes
if required_bytes > 48 * 1024:
    raise ValueError(...)
```

Both dot operands are constructed and reserved together before generated code
is mutated:

```python
self.reserve_shared_memory(lhs.nbytes + rhs.nbytes)
self.append_shared_buffer_declaration(lhs)
self.append_shared_buffer_declaration(rhs)
```

That order makes failure atomic from the code generator's perspective. If the
pair does not fit, it leaves no half-emitted declaration or cooperative loop.

Reduction scratch buffers now go through the same accounting path, so two
backend features cannot silently overcommit the block's shared memory.

## Staging inside a runtime K loop

The real target is a tiled matmul:

```python
acc = tl.zeros((BM, BN), tl.float32)

for k_base in range(0, K, BK):
    # build A [BM, BK] and B [BK, BN] loads
    acc = acc + tl.dot(a_values, b_values)
```

`SSADefinitions` recursively indexes loop bodies, so the matcher can find the
loads and dot inside an `SSAForRange`. The recovered plan keeps the loop index
as an SSA operand:

```text
A column_offset = loop.index
B row_offset    = loop.index
```

When CUDA lowering enters the loop, that operand resolves to the generated
`k_base` variable. Shared declarations remain at kernel scope while the copy
loops are emitted inside the runtime loop body.

This is the first point where structured regions from Part 10 and CUDA tile
layouts from Part 12 directly support the same feature.

## An intentionally incomplete CUDA fragment

For a recognized dot, Version 14 emits code shaped like:

```c++
__shared__ float dot_lhs_29[64];
__shared__ float dot_rhs_29[128];

// cooperative A load
for (...) {
    ...
}

// cooperative B load
for (...) {
    ...
}

__syncthreads();
```

Compilation then stops with:

```text
CUDA shared-memory staging for tl.dot is implemented,
but CUDA computation for tl.dot is not implemented
```

That diagnostic draws a precise line around the milestone. The compiler has
proven it can find and stage both operands. It has not yet claimed to compute
their product.

Dots whose operands are not canonical loads retain the more general error:

```text
CUDA lowering for tl.dot is not implemented
```

For example, the `tl.zeros` operands used in Part 13 have valid `dot` semantics
but no global matrix tiles to stage.

## Tests as structural documentation

The Version 14 tests cover the feature from small pieces to the AST frontend.

Buffer tests check rank, size, byte count, and row-major indexing.

Code-generation tests assert exact shared declarations, cooperative loop
coordinates, bounds, zero fill, and barrier placement.

Matcher tests recover the expected `CudaGlobalTilePlan` for both `A` and `B`.
They also reject malformed bounds and non-canonical operands.

Analysis tests verify the exact set of staging-only SSA IDs and, separately,
prove that dependencies shared with output address calculation are preserved.

Runtime-loop tests check that the loop induction value becomes `A`'s column
origin and `B`'s row origin. Budget tests verify that an oversized pair fails
before emitting partial code.

Finally, an AST frontend test compiles the original kernel down to the expected
Version 14 diagnostic. The failure is expected; reaching it proves that all
earlier stages agreed on the recognized tile structure.

## What Version 14 does not do

Version 14 does not:

- read the staged tiles to compute a result;
- emit a CUDA-core FMA loop;
- reuse the shared buffers across K tiles safely after computation;
- recognize arbitrary pointer expressions or load masks;
- support non-row-major cooperative copies;
- pad or swizzle shared-memory layouts;
- use vectorized or asynchronous loads;
- use tensor cores.

The shared buffers are a correct, explicit intermediate destination, not yet a
complete matmul implementation.

## What changed conceptually

Before Version 14, `dot` operands were only typed SSA block values:

```text
SSA load graph -> block<BMxBK>, block<BKxBN> -> dot
```

After Version 14, the CUDA backend can reinterpret a supported load graph as a
physical data-movement plan:

```text
SSA dot operands
       |
       v
canonical-pattern matching
       |
       v
global tile plans
       |
       v
cooperative masked copies
       |
       v
shared A and shared B + barrier
```

This is the bridge from logical tiles to CUDA's memory hierarchy. Computation
is still missing, but its inputs are now in the right place and protected by an
explicit synchronization contract.

All code for this milestone is available at
[https://github.com/pbelevich/mytriton/tree/ver14](https://github.com/pbelevich/mytriton/tree/ver14).

Next: [Part 15: CUDA-Core tl.dot]({% post_url 2026-08-30-My_Triton_From_Scratch_Part_15_CUDA_Core_Dot %}).
