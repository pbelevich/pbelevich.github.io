---
layout: post
title:  "My Triton From Scratch Part 15: CUDA-Core tl.dot"
date:   2026-08-30 16:00:00 +0000
# categories:
---

In [Part 10: Runtime For Loops]({% post_url 2026-08-08-My_Triton_From_Scratch_Part_10_Runtime_For_Loops %}),
mytriton learned to carry an accumulator through a runtime K loop.

In [Part 11: Block Factory Functions]({% post_url 2026-08-15-My_Triton_From_Scratch_Part_11_Block_Factory_Functions %}),
that accumulator became an explicit `tl.zeros` block.

In [Part 12: CUDA Tile Layouts]({% post_url 2026-08-22-My_Triton_From_Scratch_Part_12_CUDA_Tile_Layouts %}),
logical tiles were separated from their physical thread organization.

In [Part 13: tl.dot Semantics]({% post_url 2026-08-23-My_Triton_From_Scratch_Part_13_Dot_Semantics %}),
matrix multiplication became a typed and verified SSA operation.

In [Part 14: Shared-Memory Tiles]({% post_url 2026-08-29-My_Triton_From_Scratch_Part_14_Shared_Memory_Tiles %}),
the CUDA backend learned to recognize canonical matrix loads and stage their
`A[BM, BK]` and `B[BK, BN]` tiles cooperatively in shared memory.

At the end of Part 14, generated CUDA reaches this state:

```text
shared A is populated
shared B is populated
all threads passed __syncthreads()
```

Version 15 completes the first working `tl.dot` lowering. It uses ordinary
CUDA cores, one output element per thread, and a scalar `f32` accumulator. The
goal is correctness and a clear execution model, not peak performance.

## The smallest correct computation

For one output tile:

```text
A: [BM, BK]
B: [BK, BN]
C: [BM, BN]
```

Version 15 launches `BM * BN` threads. A linear thread ID is decoded into one
output coordinate:

```c++
int tile_i = threadIdx.x / BN;
int tile_j = threadIdx.x % BN;
```

That thread owns exactly one value:

```text
C[tile_i, tile_j]
```

It computes the inner product of one shared-memory row of `A` and one
shared-memory column of `B`:

```text
acc = 0
for k in [0, BK):
    acc += shared_A[tile_i, k] * shared_B[k, tile_j]
```

This is not a tensor-core operation. It is an ordinary loop of scalar
floating-point multiplies and adds compiled for CUDA cores.

## Validating the lowering boundary

`emit_dot_from_shared_memory` receives a typed SSA result and the two shared
buffers created by Version 14. Before generating code, it checks that the
pieces still agree.

The result must be a rank-2 block:

```python
if not isinstance(result.ty, BlockType) or result.ty.rank != 2:
    raise TypeError(...)
```

The shared operands must have matching reduction dimensions:

```python
if buffers.lhs.columns != buffers.rhs.rows:
    raise TypeError(...)
```

Their outer dimensions determine the exact output shape:

```python
expected_shape = (
    buffers.lhs.rows,
    buffers.rhs.columns,
)
```

The initial execution model also requires one thread per output element:

```python
if result.ty.shape != self.layout.thread_shape:
    raise TypeError(
        "CUDA-core dot currently requires one CUDA thread per "
        "result element"
    )
```

Finally, all three element types must be `f32`.

Some of these facts were already established by type inference and SSA
verification. The backend checks its own assumptions anyway because it also
combines information from CUDA layouts and physical shared buffers.

## Emitting the accumulator

The implementation derives the shared-memory expressions from the current
thread coordinate:

```python
row = self.thread_coordinate(0)
column = self.thread_coordinate(1)

lhs_element = buffers.lhs.element(row, reduction_index)
rhs_element = buffers.rhs.element(reduction_index, column)
```

Then it emits one local variable and one reduction loop:

```c++
float v29 = 0.0f;
for (int dot_k_29 = 0; dot_k_29 < 16; ++dot_k_29) {
    v29 +=
        dot_lhs_29[(tile_i) * 16 + (dot_k_29)] *
        dot_rhs_29[(dot_k_29) * 8 + (tile_j)];
}
```

The SSA result `%29` is recorded as CUDA variable `v29`, so later operations
can consume it normally:

```text
%30 = add %acc, %29
```

becomes:

```c++
float v30 = (acc + v29);
```

The special behavior is confined to producing the dot result. Once produced,
it participates in the existing scalar-per-thread CUDA lowering.

## The complete single-tile phase order

Combining Versions 14 and 15 gives this sequence:

```text
cooperatively load A into shared memory
cooperatively load B into shared memory
                  |
                  v
           __syncthreads()
                  |
                  v
each thread computes one dot product
                  |
                  v
           __syncthreads()
```

The generated core looks like:

```c++
__shared__ float dot_lhs_29[BM * BK];
__shared__ float dot_rhs_29[BK * BN];

for (int index = threadIdx.x;
     index < BM * BK;
     index += BM * BN) {
    // masked global A -> shared A
}

for (int index = threadIdx.x;
     index < BK * BN;
     index += BM * BN) {
    // masked global B -> shared B
}

__syncthreads();

float result = 0.0f;
for (int k = 0; k < BK; ++k) {
    result += shared_A[tile_i * BK + k]
            * shared_B[k * BN + tile_j];
}

__syncthreads();
```

The first barrier protects reads. The second protects reuse.

## Why there are two barriers

The first `__syncthreads()` has an obvious purpose: every thread must finish
writing its assigned shared locations before any thread starts the FMA loop.

The second barrier can look redundant for a kernel that computes only one
K-tile and exits. It is required for the real runtime loop:

```python
for k_base in range(0, K, BK):
    a_values = tl.load(...)
    b_values = tl.load(...)
    acc = acc + tl.dot(a_values, b_values)
```

Shared buffer names are allocated once at kernel scope and reused by every loop
iteration. Without a barrier after computation, a fast thread could start
overwriting the next `A` or `B` tile while a slower thread was still reading
the current one.

The loop's synchronization protocol is therefore:

```text
iteration t:
    write shared A/B
    barrier: writes complete
    read shared A/B and compute
    barrier: reads complete

iteration t + 1:
    overwrite shared A/B safely
```

Barriers describe ownership transitions of the same memory, not merely pauses
in execution.

## Accumulating several K tiles

A complete matrix product often has `K > BK`. The AST kernel uses the structured
loop introduced in Part 10:

```python
acc = tl.zeros((BM, BN), tl.float32)

for k_base in range(0, K, BK):
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

    acc = acc + tl.dot(a_values, b_values)
```

The SSA loop carries `acc` from one iteration to the next:

```text
%result = for %k_base in range(0, K, BK)
          iter_args(%acc_arg = %zero) : block<BMxBN x f32> {
    ...
    %dot = dot %a_values, %b_values
    %next = add %acc_arg, %dot
    yield %next
}
```

In CUDA, the loop contains both cooperative loads, both barriers, the FMA loop,
and the accumulator update. The output store remains after the loop.

This connects all of the earlier pieces:

```text
AST runtime loop
    +-- loop-carried tl.zeros accumulator
    +-- canonical rank-2 loads
    +-- tl.dot SSA operation
    +-- cooperative shared-memory staging
    +-- CUDA-core computation
```

## Partial K tiles are already handled

Suppose `K = 19` and `BK = 8`. The runtime loop visits:

```text
k_base = 0
k_base = 8
k_base = 16
```

The final logical tile covers columns or rows `16..23`, but only `16..18` are
valid. Version 14's masks write `0.0f` into all out-of-bounds shared positions.

The Version 15 FMA loop can still execute exactly `BK` iterations:

```text
valid products + zero-padded products
```

No special tail loop is necessary, and all threads follow the same control
flow around both barriers.

The same mechanism protects partial `M` and `N` tiles. Input loads are padded
and the final C store uses its existing output mask.

## One output element per thread

For `BM = 4` and `BN = 8`, the output tile has 32 elements and the kernel uses
32 threads:

```text
thread 0  -> C[0, 0]
thread 1  -> C[0, 1]
...
thread 8  -> C[1, 0]
...
thread 31 -> C[3, 7]
```

Every thread owns one scalar `v29`. Existing pointer arithmetic, masks, loop
carried values, and stores therefore continue to work without a new value
representation.

That simplicity is the main reason to begin here. It makes the shared-memory
protocol and numerical result easy to validate before introducing multiple
registers per thread.

It is also a serious limitation. A `16 × 16` result would require 256 threads
and still give each thread only one accumulator. The backend cannot yet choose
a smaller thread tile and let one thread compute several C values.

## Exact generated CUDA is a compiler test

One test lowers a canonical `[4, 16] × [16, 8]` dot and compares the complete
CUDA source. The key section is:

```c++
__syncthreads();
float v29 = 0.0f;
for (int dot_k_29 = 0; dot_k_29 < 16; ++dot_k_29) {
    v29 += dot_lhs_29[(tile_i) * 16 + (dot_k_29)]
         * dot_rhs_29[(dot_k_29) * 8 + (tile_j)];
}
__syncthreads();
```

An exact-source test catches details that numerical tests might miss:

- shared declarations are at kernel scope;
- staging precedes computation;
- the row and column indices are not transposed;
- the loop bound is `BK`;
- both barriers occur in the required order;
- the result is connected to the final store.

There is also a structural runtime-loop test. Rather than comparing one very
large string, it checks the relative positions of:

```text
outer K loop
< A cooperative load
< B cooperative load
< first barrier
< FMA loop
< second barrier
< accumulator update
```

This documents the synchronization protocol directly.

## Executing against a reference matmul

Version 15 adds CUDA execution tests through CuPy.

The single-tile case uses dimensions that exercise masked M, N, and K edges:

```text
M = 7, N = 13, K = 7
BM = 4, BN = 8, BK = 8
```

The grid contains several output blocks, and the K tile is partially filled.
The generated result is compared with:

```python
expected = a @ b
cp.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)
```

The multiple-tile case uses:

```text
M = 7, N = 13, K = 19
BM = 4, BN = 8, BK = 8
```

That covers three K-loop iterations, shared-buffer reuse, and a partial final
tile. Passing it gives much stronger evidence than a single exact CUDA string:
the whole path from Python AST to executing GPU code agrees with matrix
multiplication semantics.

## Honest performance boundaries

The implementation is tiled, but it is not yet a fast Triton-style matmul.

It has:

- cooperative global-to-shared copies;
- shared-memory reuse;
- correct masked edge handling;
- a runtime K-tile loop;
- ordinary CUDA-core multiply-adds.

It does not have:

- several output accumulators per thread;
- vectorized global loads;
- deliberate coalescing analysis;
- shared-memory padding or swizzling;
- double buffering;
- asynchronous copies;
- mixed-precision inputs;
- tensor-core instructions;
- autotuning.

The fixed one-result-per-thread policy is the next limitation to remove. Until
then, larger C tiles directly imply larger CUDA thread blocks.

## What changed conceptually

At the end of Version 14, the backend had prepared data but refused to claim a
result:

```text
global tiles -> shared tiles -> barrier -> explicit missing-compute error
```

Version 15 completes the first honest execution path:

```text
global A/B
    |
    v
cooperative shared-memory staging
    |
    v
barrier
    |
    v
one scalar FMA reduction per output thread
    |
    v
barrier before shared-buffer reuse
    |
    v
loop-carried accumulation and masked C store
```

`tl.dot` is no longer just a frontend and IR promise. For canonical `f32`
matrix tiles, it now produces the right answer on a GPU.

All code for this milestone is available at
[https://github.com/pbelevich/mytriton/tree/ver15](https://github.com/pbelevich/mytriton/tree/ver15).

The next milestone will let one CUDA thread own several output values in a
register tile. That post is not published yet.
