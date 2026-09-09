---
layout: post
title:  "My Triton From Scratch Part 18: Shared-Memory Optimization"
date:   2026-09-07 16:00:00 +0000
# categories:
---

In [Part 14: Shared-Memory Tiles]({% post_url 2026-08-29-My_Triton_From_Scratch_Part_14_Shared_Memory_Tiles %}),
mytriton learned to stage canonical matrix tiles from global memory into shared
memory.

In [Part 15: CUDA-Core tl.dot]({% post_url 2026-08-30-My_Triton_From_Scratch_Part_15_CUDA_Core_Dot %}),
each CUDA thread consumed those shared tiles in a scalar FMA loop.

In [Part 16: Register Tiles]({% post_url 2026-09-05-My_Triton_From_Scratch_Part_16_Register_Tiles %}),
one thread began computing several output elements in registers.

In [Part 17: PyTorch Tensor Interoperability]({% post_url 2026-09-06-My_Triton_From_Scratch_Part_17_PyTorch_Tensor_Interoperability %}),
the same kernels gained a zero-copy path to PyTorch tensors.

The matrix multiplication is correct, but its shared-memory layout and
synchronization policy are still the simplest possible ones:

```text
flat row-major A and B buffers
two barriers per K tile
ordinary runtime loop over BK
```

Version 18 makes those physical choices explicit and improves them in three
small steps:

1. pad shared-memory rows when the current access pattern would create bank
   conflicts;
2. ask the CUDA compiler to unroll the fixed-size inner dot loop;
3. use two shared-memory stages for canonical runtime K loops so consecutive
   iterations alternate between separate storage.

This is an optimization milestone, but it remains deliberately conservative.
Every optimization is guarded by a structural contract, and unsupported loop
shapes keep the older correct lowering.

## The Version 17 phase order

Before this change, every K-tile iteration follows the same protocol:

```text
cooperatively write A and B
              |
              v
       __syncthreads()
              |
              v
read A and B in the FMA loop
              |
              v
       __syncthreads()
```

The first barrier prevents an early reader from observing an incomplete tile.
The second prevents the next iteration from overwriting the same shared arrays
while another thread is still reading them.

That protocol is correct, but the physical arrays have no room for alternative
layouts or generations of data:

```python
size = rows * columns
index = row * columns + column
```

Version 18 begins by separating logical shape from physical storage.

## A shared buffer needs a physical row stride

`CudaSharedBuffer` now records two new parameters:

```python
@dataclass(frozen=True)
class CudaSharedBuffer:
    name: str
    logical_shape: tuple[int, ...]
    element_ty: ScalarType
    row_padding: int = 0
    stage_count: int = 1
```

Its logical dimensions do not change. A dot still consumes an `A[BM, BK]`
tile or a `B[BK, BN]` tile. The backend derives a physical row stride:

```text
row_stride = columns + row_padding
```

One copy of the logical tile occupies:

```text
stage_size = rows * row_stride
```

The complete allocation occupies:

```text
size = stage_count * stage_size
```

An element address becomes:

```text
stage * stage_size + row * row_stride + column
```

This gives three independent coordinate spaces:

```text
logical row/column -> meaning of the matrix tile
physical row stride -> spacing between shared rows
stage               -> which temporal copy of the tile is active
```

The dot semantics remain unchanged while the backend is free to change the
storage layout underneath them.

## Why shared-memory banks matter

CUDA shared memory is divided into 32 banks. For 32-bit values, consecutive
`f32` words map to consecutive banks:

```text
bank = word_offset % 32
```

When threads in a warp access different addresses in the same bank, the access
may be serialized into several transactions. A broadcast of exactly the same
address is a special case, but unrelated addresses that collide are a real
cost.

The CUDA-core dot path reads A by output row:

```text
A[logical_output_row, k]
```

Several active rows may therefore read the same `k` column at once. With an
unpadded row stride of 16, the bank used by column zero is:

```text
row 0 -> bank 0
row 1 -> bank 16
row 2 -> bank 0
row 3 -> bank 16
```

Four rows reach only two banks. Rows zero and two collide; rows one and three
collide.

Adding one unused element to every row changes the stride to 17:

```text
row 0 -> bank 0
row 1 -> bank 17
row 2 -> bank 2
row 3 -> bank 19
```

Now all four rows reach distinct banks for the same logical column.

The matrix still has 16 logical columns. Column 16 is not part of the tile; it
is only physical padding that rotates the next row's bank mapping.

## Choosing padding instead of hard-coding it

The backend computes how many banks a row stride can reach before repeating:

```python
distinct_banks = 32 // gcd(columns, 32)
```

If the CUDA thread layout reads more rows simultaneously than that number, one
padding element is useful:

```python
return int(simultaneous_rows > distinct_banks)
```

Some examples are:

```text
columns=16, simultaneous_rows=4 -> padding=1
columns=16, simultaneous_rows=2 -> padding=0
columns=8,  simultaneous_rows=4 -> padding=0
columns=8,  simultaneous_rows=8 -> padding=1
columns=32, simultaneous_rows=2 -> padding=1
columns=7,  simultaneous_rows=4 -> padding=0
```

This is not a complete bank-conflict optimizer. It models the specific A-tile
read pattern generated by the current one-warp register layout.

Only A receives this padding. The present B access is shared across output
rows in a way that acts as a broadcast, so applying the same rule to B would
spend shared memory without addressing the same conflict.

## Every producer and consumer must use the new stride

Padding is only correct if both sides agree on the physical address.

The cooperative load writes:

```c++
shared_A[row * 17 + column] = in_bounds
    ? global_A[global_row * K + global_column]
    : 0.0f;
```

The FMA loop reads:

```c++
shared_A[tile_i * 17 + dot_k]
```

If the writer used 17 and the reader still used 16, the kernel would compile
and return incorrect values. Centralizing address construction in
`CudaSharedBuffer.element()` keeps the row stride shared by staging and
computation.

The shared-memory budget also uses the physical size:

```text
nbytes = stage_count * rows * row_stride * element_size
```

Padding is therefore visible to resource validation rather than being hidden
inside string generation.

## Unrolling the inner reduction loop

The dot reduction width `BK` is a compile-time tile dimension. The generated
loop still had ordinary runtime syntax:

```c++
for (int dot_k = 0; dot_k < BK; ++dot_k) {
    accumulator += shared_A[...] * shared_B[...];
}
```

Version 18 emits:

```c++
#pragma unroll
for (int dot_k = 0; dot_k < BK; ++dot_k) {
    accumulator += shared_A[...] * shared_B[...];
}
```

The source remains compact, while the CUDA compiler is told that the fixed
trip count should be expanded. That removes loop-control overhead and exposes
independent accumulator updates and address expressions to later compiler
optimization.

This is a backend hint, not a new SSA transformation. The logical `dot`
operation and the runtime outer K-tile loop remain unchanged.

## One stage versus two stages

A *stage* is one physical copy of a logical shared tile.

With one stage:

```text
shared_A = [A tile]
shared_B = [B tile]
```

Every outer-loop iteration overwrites those same addresses.

With two stages:

```text
shared_A = [A tile stage 0][A tile stage 1]
shared_B = [B tile stage 0][B tile stage 1]
```

The logical tile shapes are still `[BM, BK]` and `[BK, BN]`. `stage_count=2`
only doubles their physical storage.

Iteration `k_base` selects a stage with:

```c++
int dot_stage = (k_base / BK) & 1;
```

For a loop with `BK = 8`, the mapping is:

```text
k_base = 0  -> stage 0
k_base = 8  -> stage 1
k_base = 16 -> stage 0
k_base = 24 -> stage 1
```

This is a ping-pong buffer.

## What ping-pong changes about synchronization

With one stage, iteration `t + 1` overwrites exactly the storage read by
iteration `t`. A barrier after the dot protects that reuse.

With two stages, iteration `t + 1` writes the other storage region:

```text
iteration t:     read stage 0
iteration t + 1: write stage 1
```

The trailing reuse barrier can be omitted for the canonical loop. The load
barrier inside the next iteration still ensures that its complete A and B tiles
are visible before computation begins.

The generated phase order becomes:

```text
select stage
cooperatively write that stage
              |
              v
       __syncthreads()
              |
              v
read that stage and compute
              |
              v
advance to the other stage
```

One barrier per iteration remains instead of two.

## Two stages do not yet mean overlap

It is tempting to call this a fully pipelined matmul, but that would overstate
the implementation.

Version 18 still performs operations sequentially:

```text
load current tile
barrier
compute current tile
load next tile
barrier
compute next tile
```

It does not prefetch tile `t + 1` while computing tile `t`. The cooperative
loads are synchronous, and there is no `cp.async`, producer/consumer barrier,
or software pipeline.

The second stage currently provides two concrete benefits:

```text
different storage for consecutive iterations
one fewer block barrier per iteration
```

It also establishes the representation a later asynchronous pipeline can use.

## Matching the loop before changing it

Double buffering is not applied to every `SSAForRange`. The backend recognizes
one canonical matmul structure.

The loop must:

- start at integer zero;
- use a positive compile-time step;
- advance by the dot reduction size `BK`;
- contain exactly one stageable dot;
- use the loop index as A's column offset;
- use the loop index as B's row offset;
- carry exactly one accumulator;
- add the dot result to that carried accumulator;
- yield exactly the addition result;
- contain no nested runtime loop.

In simplified SSA:

```text
%result = for %k in range(0, K, BK)
          iter_args(%acc = %zero) {
    ... canonical A/B loads using %k ...
    %dot = dot %a, %b
    %next = add %acc, %dot
    yield %next
}
```

The matcher returns a small plan containing the dot result ID and
`stage_count=2`. Code generation then uses the same stage expression for
cooperative writes and dot reads.

## Why a conservative matcher matters

An optimization is only valid for the structure whose dependencies it
understands.

Suppose a loop contains another operation, carries another value, advances by
half of `BK`, starts from a nonzero offset, or nests another runtime loop.
Silently applying the ping-pong lowering could select the wrong stage or remove
a barrier that protects another use.

Version 18 responds by declining the optimization:

```text
match succeeds -> two stages, one barrier
match fails    -> one stage, original two barriers
```

The fallback is not an error. It is the existing correct implementation.
This is a useful compiler pattern: specialized lowering should be optional
unless the IR satisfies its complete proof obligation.

## Stage selection belongs inside the runtime loop

Shared arrays are declared once at kernel scope:

```c++
__shared__ float dot_lhs[2 * lhs_stage_size];
__shared__ float dot_rhs[2 * rhs_stage_size];
```

The stage expression depends on the runtime loop induction variable, so it is
emitted inside the loop:

```c++
for (int k_base = 0; k_base < K; k_base += BK) {
    int dot_stage = (k_base / BK) & 1;

    // cooperative writes into dot_stage
    __syncthreads();
    // FMA reads from the same dot_stage
}
```

This uses the structured loop representation introduced much earlier. The
backend can distinguish kernel-scope storage from region-scope values without
flattening the loop into an opaque string first.

## Shared-memory budget checks include stages and padding

The backend retains its conservative 48 KiB per-block limit.

For a two-stage dot, the required bytes are:

```text
2 * (
    BM * (BK + lhs_padding)
    + BK * BN
) * sizeof(f32)
```

The allocation pair is reserved before either declaration or copy loop is
emitted. If it exceeds the budget, code generation fails without leaving a
half-generated kernel.

This check becomes more important after Version 18 because both optimizations
consume extra storage:

```text
row padding -> a few additional words per A row
two stages  -> twice the complete A/B allocation
```

An optimization that ignores its resource cost is not a complete backend
decision.

## Tests treat physical layout as a contract

Buffer tests verify:

- logical dimensions;
- padded row strides;
- stage sizes and total allocation sizes;
- exact integer and symbolic offsets;
- invalid padding, stages, and coordinates;
- matching stage counts for A and B.

Bank tests check the exact before-and-after mapping:

```text
unpadded banks = (0, 16, 0, 16)
padded banks   = (0, 17, 2, 19)
```

Code-generation tests assert that cooperative stores and FMA loads use the
same stage and padded stride. They also verify `#pragma unroll` and the absence
of the trailing barrier in the matched path.

Matcher tests mutate one condition at a time: loop start, step, accumulator
operation, yield, or nesting. Every invalid canonical form must return to the
single-stage implementation.

CUDA execution tests continue to compare the complete matmul against a
reference across multiple K tiles and partial edges. The optimization is only
successful if it preserves all of the correctness established in Parts 15 and
16.

## What Version 18 does not do

Version 18 does not:

- issue asynchronous shared-memory copies;
- overlap tile loading with dot computation;
- vectorize global loads;
- swizzle shared-memory coordinates;
- tune padding for arbitrary access patterns;
- use more than the current one-warp dot layout;
- introduce low-precision inputs;
- emit tensor-core instructions.

The name “double buffering” describes the two physical stages, not a completed
asynchronous pipeline.

## What changed conceptually

Before Version 18, a shared tile had only a logical shape:

```text
[rows, columns] -> one flat allocation
```

After Version 18, a shared tile has a small physical layout:

```text
logical [rows, columns]
          |
          +--> row_stride = columns + padding
          |
          +--> stage_size = rows * row_stride
          |
          +--> stage_count
          |
          v
physical shared-memory address
```

That representation supports both spatial optimization, through bank-aware
row padding, and temporal organization, through ping-pong stages.

The optimization pipeline is still honest about what it proves:

```text
canonical loop -> specialize
general loop   -> preserve the known-correct fallback
```

All code for this milestone is available at
[https://github.com/pbelevich/mytriton/tree/ver18](https://github.com/pbelevich/mytriton/tree/ver18).

The next milestone adds `float16` and `bfloat16`, explicit casts, shared
numeric-promotion rules, and `f32` accumulation for low-precision `tl.dot`.
