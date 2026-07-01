---
layout: post
title:  "My Triton Language From Scratch Part 6"
date:   2026-06-27 15:00:00 +0000
# categories:
---

In [Part 1]({% post_url 2026-06-22-My_Triton_Language_From_Scratch_-_Part_1 %}),
mytriton learned how to trace a Python function into an expression-tree IR.

In [Part 2]({% post_url 2026-06-23-My_Triton_Language_From_Scratch_-_Part_2 %}),
it learned how to infer types and lower that tree into typed SSA-style
operations.

In [Part 3]({% post_url 2026-06-24-My_Triton_Language_From_Scratch_-_Part_3 %}),
it learned how to turn SSA into CUDA C++ and launch the generated kernel.

In [Part 4]({% post_url 2026-06-25-My_Triton_Language_From_Scratch_-_Part_4 %}),
the language grew enough elementwise operations to write ReLU, leaky ReLU, and
sigmoid.

In [Part 5]({% post_url 2026-06-26-My_Triton_Language_From_Scratch_-_Part_5 %}),
the middle of the compiler became stricter: SSA is now verified, optimized, and
verified again before CUDA code generation.

Version 6 is the first version where vectors stop being only independent
per-lane values.

Until now, a `vector<256 x f32>` mostly meant "256 unrelated scalar operations,
one per CUDA thread." That is enough for elementwise kernels. Each lane loads
one value, computes one value, and stores one value.

Reductions are different. A row-wise sum, max, min, or softmax needs the lanes
inside a block to cooperate. Each thread still owns one lane, but now the block
also has to combine those lanes into one scalar result.

This version adds exactly that first cooperative step:

- a real two-dimensional matrix kernel using `tl.program_id(0)` and
  `tl.program_id(1)`;
- row-wise `tl.sum`, `tl.max`, and `tl.min`;
- shared-memory CUDA lowering for block-local reductions;
- a numerically stable row-wise softmax;
- `tl.static_range` for compile-time loop unrolling;
- a first naive matrix multiplication kernel;
- a long-row sum that handles rows wider than one CUDA block.

The code for this milestone is here:
[https://github.com/pbelevich/mytriton/tree/ver6](https://github.com/pbelevich/mytriton/tree/ver6).

The verifier and tests grew too, but they are not the story of this part. The
interesting idea is how a vector value becomes a block-local reduction domain.

## A small detour: using the second program axis

Before reductions, Version 6 adds one useful bit of grid vocabulary: a matrix
kernel that uses two program IDs.

The kernel is still elementwise, but the launch grid is two-dimensional:

```python
@triton.jit
def matrix_add_kernel(x, y, out, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col_block = tl.program_id(1)

    cols = col_block * BLOCK + tl.arange(0, BLOCK)

    offsets = row * n_cols + cols
    mask = cols < n_cols

    lhs = tl.load(x + offsets, mask=mask, other=0.0)
    rhs = tl.load(y + offsets, mask=mask, other=0.0)

    tl.store(out + offsets, lhs + rhs, mask=mask)
```

The first program axis chooses the row. The second chooses a block of columns.
The launch grid is:

```python
lambda meta: (rows, triton.cdiv(cols, meta["BLOCK"]))
```

That gives one CUDA block per `(row, column-block)` tile.

The SSA makes both axes explicit:

```text
%0 = program_id {axis=0} : i32
%1 = mul %0, n_cols : i32
%2 = program_id {axis=1} : i32
%3 = mul %2, 256 : i32
%4 = arange {start=0, end=256} : vector<256 x i32>
%5 = add %3, %4 : vector<256 x i32>
%6 = add %1, %5 : vector<256 x i32>
```

CUDA lowering already knew how to map program axes:

```cuda
int v0 = blockIdx.x;
int v2 = blockIdx.y;
```

So this feature is not large, but it is important. It stops the language from
being only a one-dimensional vector-add toy, and it gives the reduction kernels
a natural row coordinate.

## A row-wise sum

The first reduction kernel is row-wise sum:

```python
@triton.jit
def row_sum_kernel(x, out, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    offsets = row * n_cols + cols
    mask = cols < n_cols

    values = tl.load(x + offsets, mask=mask, other=0.0)

    total = tl.sum(values)

    first_lane = cols < 1

    tl.store(out + row, total, mask=first_lane)
```

The first half should look familiar. One program instance handles one row.
`cols` is a vector of lane indices. The mask protects the final partial row
when the column count is not a power of two.

The new operation is:

```python
total = tl.sum(values)
```

`values` has vector type:

```text
vector<8 x f32>
```

but `total` is scalar:

```text
f32
```

That is the first operation in mytriton that intentionally collapses the vector
shape.

There is also a slightly odd-looking store mask:

```python
first_lane = cols < 1
tl.store(out + row, total, mask=first_lane)
```

The reduced value is the same logical scalar for the whole block, but the
current store operation is still lane-oriented. If every thread stored the same
result, the answer would probably be correct, but the kernel would have a
needless write race. Instead, only lane 0 writes the row result.

This is not the most elegant abstraction, but it keeps the language small while
making the block-local reduction visible.

## Reduction nodes

The public language functions are thin wrappers:

```python
def sum(value: Value) -> Value:
    return Value(Sum(unwrap(value)))


def max(value: Value) -> Value:
    return Value(Max(unwrap(value)))


def min(value: Value) -> Value:
    return Value(Min(unwrap(value)))
```

The expression tree gets three new node types:

```python
@dataclass
class Sum:
    value: Any


@dataclass
class Max:
    value: Any


@dataclass
class Min:
    value: Any
```

I intentionally kept these separate from `Maximum` and `Minimum`.

In mytriton, `tl.maximum(a, b)` and `tl.minimum(a, b)` are elementwise
operations. They take two inputs and preserve shape:

```text
maximum(vector<256 x f32>, f32) -> vector<256 x f32>
```

`tl.max(values)` and `tl.min(values)` are reductions. They take one vector and
produce one scalar:

```text
max(vector<256 x f32>) -> f32
min(vector<256 x f32>) -> f32
```

The names are close, but the compiler semantics are very different. Keeping
separate IR nodes makes that distinction explicit instead of smuggling it
through a flag.

SSA lowering is straightforward:

```python
if isinstance(expr, Sum):
    value = self.lower_expr(expr.value)
    return self.emit("sum", expr, operands=(value,))

if isinstance(expr, Max):
    value = self.lower_expr(expr.value)
    return self.emit("max", expr, operands=(value,))

if isinstance(expr, Min):
    value = self.lower_expr(expr.value)
    return self.emit("min", expr, operands=(value,))
```

For row-wise sum, the interesting part of the SSA is:

```text
%6 = load %4, %5, 0.0 : vector<8 x f32>
%7 = sum %6 : f32
%8 = addptr out, %0 : ptr<f32>
%9 = cmp_lt %2, 1 : vector<8 x bool>
store %8, %7, %9
```

`%6` is distributed across the block. `%7` is the scalar result of cooperating
across those lanes.

## Lowering a reduction to shared memory

The CUDA lowering for a reduction is the first backend rule that needs
communication between threads.

The current strategy is deliberately simple:

1. allocate one shared-memory array per reduction;
2. have each thread write its lane value into that array;
3. synchronize the block;
4. reduce the array by repeatedly halving the active range;
5. read the result from element 0.

For row-wise sum with width 8, CUDA lowering emits:

```cuda
__shared__ float reduce_smem_7[8];

reduce_smem_7[threadIdx.x] = v6;
__syncthreads();
for (int stride_7 = 4; stride_7 > 0; stride_7 >>= 1) {
    if (threadIdx.x < stride_7) {
        reduce_smem_7[threadIdx.x] += reduce_smem_7[threadIdx.x + stride_7];
    }
    __syncthreads();
}
float v7 = reduce_smem_7[0];
```

This is not an optimized reduction. There are no warp-level primitives, no
unrolling, no special handling for small block sizes, and no attempt to avoid
some synchronizations.

That is fine for this version. The important thing is that the lowering is
concrete and easy to inspect.

The skeleton is shared by all reductions:

{% raw %}
```python
self.shared_lines.append(f"    __shared__ {cuda_ty} {shared}[{width}];")

self.lines.extend(
    [
        f"    {shared}[threadIdx.x] = {value};",
        "    __syncthreads();",
        f"    for (int {stride} = {width // 2}; {stride} > 0; {stride} >>= 1) {{",
        f"        if (threadIdx.x < {stride}) {{",
    ]
)
```
{% endraw %}

Only the combiner changes:

```python
if opcode == "sum":
    return f"{lhs} += {rhs};"

if opcode == "max":
    return f"{lhs} = fmaxf({lhs}, {rhs});"

if opcode == "min":
    return f"{lhs} = fminf({lhs}, {rhs});"
```

I like this split because it mirrors the conceptual model:

```text
reduction = communication pattern + combine operation
```

For now, the communication pattern is always the same shared-memory tree. The
combine operation decides whether the result is a sum, maximum, or minimum.

## The verifier learns reduction legality

The verifier from
[Part 5]({% post_url 2026-06-26-My_Triton_Language_From_Scratch_-_Part_5 %})
gets one new backend-shaped rule: reductions must consume one vector whose
width matches the CUDA block size and is a power of two.

That is not a general truth about reductions. It is a truth about this backend.
The current lowering halves the active range:

```cuda
for (int stride = width / 2; stride > 0; stride >>= 1)
```

That pattern assumes a power-of-two width. It also assumes that every vector
lane corresponds to one CUDA thread in the block.

So the verifier checks it before codegen:

```python
if not isinstance(value_ty, VectorType):
    self.fail(index, op, f"reduction expects vector, got {value_ty}")

if value_ty.size != self.block_size:
    self.fail(index, op, "reduction width does not match CUDA block size")

if value_ty.size & (value_ty.size - 1):
    self.fail(index, op, "reduction width must be a power of two")

self.require_type(index, op, result_ty, value_ty.element)
```

This is exactly the kind of invariant I want the verifier to carry. CUDA
codegen can stay simple because the IR already promised to be legal for the
lowering strategy.

## Row-wise max and min

Once the reduction skeleton exists, max and min are almost the same kernel as
sum.

Row-wise max uses a different fallback value:

```python
values = tl.load(x + offsets, mask=mask, other=float("-inf"))

total = tl.max(values)
```

Masked-off lanes should not affect the maximum, so they load negative infinity.

Row-wise min uses positive infinity:

```python
values = tl.load(x + offsets, mask=mask, other=float("inf"))

total = tl.min(values)
```

The generated CUDA changes the combine line:

```cuda
reduce_smem_7[threadIdx.x] =
    fmaxf(reduce_smem_7[threadIdx.x],
          reduce_smem_7[threadIdx.x + stride_7]);
```

or:

```cuda
reduce_smem_7[threadIdx.x] =
    fminf(reduce_smem_7[threadIdx.x],
          reduce_smem_7[threadIdx.x + stride_7]);
```

The important point is that there is still no row-max-specific backend pass.
There is a reduction opcode and a combiner. That is enough to express the
kernel.

## Softmax is two reductions and some elementwise math

The main kernel I wanted out of Version 6 is row-wise softmax:

```python
@triton.jit
def softmax_kernel(x, out, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    offsets = row * n_cols + cols
    mask = cols < n_cols

    values = tl.load(
        x + offsets,
        mask=mask,
        other=float("-inf"),
    )

    row_max = tl.max(values)
    numerators = tl.exp(values - row_max)
    denominator = tl.sum(numerators)
    probabilities = numerators / denominator

    tl.store(out + offsets, probabilities, mask=mask)
```

This is the first mytriton kernel that feels like a small real GPU primitive.
It uses:

- masked loads;
- a max reduction;
- scalar-vector broadcasting;
- `tl.exp`;
- a sum reduction;
- division;
- a masked store.

There is no special `softmax` operation. The compiler only knows the pieces.

The loaded row values are a vector:

```text
%6 = load %4, %5, -inf : vector<8 x f32>
```

The row maximum is scalar:

```text
%7 = max %6 : f32
```

Subtracting it from the vector broadcasts the scalar across lanes:

```text
%8 = sub %6, %7 : vector<8 x f32>
```

Then `exp` preserves the vector shape:

```text
%9 = exp %8 : vector<8 x f32>
```

The denominator is another scalar reduction:

```text
%10 = sum %9 : f32
```

Dividing the vector of numerators by the scalar denominator produces the final
probabilities:

```text
%11 = div %9, %10 : vector<8 x f32>
```

That sequence is exactly why I wanted reductions to return scalars rather than
fake vector values. The scalar result naturally broadcasts when it meets a
vector operation later.

The `row_max` subtraction also makes the softmax numerically stable. A row like:

```text
[1000, 1001, 999]
```

does not try to compute `exp(1001)` directly. The largest value is subtracted
first, so the exponentials operate on values around zero:

```text
[-1, 0, -2]
```

That stability comes from the source kernel, but the compiler has to preserve
the shape changes correctly for the formula to work.

## Multiple reductions in one block

Softmax also exercises a new backend case: two reductions in one kernel.

CUDA lowering allocates one shared-memory buffer for the max:

```cuda
__shared__ float reduce_smem_7[8];
```

and another for the sum:

```cuda
__shared__ float reduce_smem_10[8];
```

Each buffer name includes the SSA result ID. That is a very simple naming
scheme, but it avoids collisions and makes the generated source easy to relate
back to the SSA.

The generated CUDA is not trying to minimize shared-memory usage yet. A later
backend could reuse the same shared array when lifetimes do not overlap. For
now, clarity wins.

## Shared memory is still internal

The reduction lowering uses shared memory internally, but Version 6 does not
expose shared-memory allocation in the source language.

That distinction matters. A row-wise sum can be represented as one pure-looking
SSA operation:

```text
%7 = sum %6 : f32
```

The CUDA backend is free to lower that operation with scratch storage and
thread synchronization because the whole communication pattern is owned by the
reduction primitive.

Tiled matrix multiplication is different. It needs source-visible intermediate
state: load a tile, write it somewhere close to the block, synchronize, read it
many times, then move to the next tile. mytriton cannot express that yet. That
is why the matmul in this part stays deliberately naive.

## The one-block row limitation

At this point, row-wise sum, max, min, and softmax all share a limitation:

```text
one row == one CUDA block
one column == one lane/thread
```

That is fine for rows up to 1024 columns, because CUDA blocks can have up to
1024 threads in this backend. But it is not enough for longer rows.

There are two broad ways forward:

1. use multiple blocks per row and reduce partial results across blocks;
2. keep one block per row, but let each thread process more than one column.

Version 6 takes the second path first because it needs only one new language
idea: compile-time loops. The same idea also makes it possible to write the
first, deliberately naive, matrix multiplication kernel.

## `tl.static_range`

`tl.static_range` is not a runtime loop in the IR. It is a Python range that is
validated to be compile-time-known:

```python
def static_range(start: int, stop: int | None = None, step: int = 1) -> range:
    if stop is None:
        start, stop = 0, start

    for name, value in (
        ("start", start),
        ("stop", stop),
        ("step", step),
    ):
        if not isinstance(value, int):
            raise ValueError(
                f"static_range {name} must be compile-time int, "
                f"got {type(value).__name__}"
            )

    return range(start, stop, step)
```

When tracing runs the kernel, Python executes the loop immediately. Every
iteration adds more expression nodes. There is no loop opcode yet.

That makes `static_range` a good fit for the current straight-line IR. It gives
the source language a loop-shaped syntax while keeping the compiler output as
ordinary unrolled SSA.

## A first naive matmul

With a two-dimensional launch grid and compile-time loops, mytriton can now
write a first matrix multiplication kernel. This is not the tiled matmul I want
long term. It does not use shared memory to cache tiles. It simply computes one
row and a block of output columns at a time:

```python
@triton.jit
def naive_matmul_kernel(a, b, c, n_cols, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col_block = tl.program_id(1)

    cols = col_block * BLOCK + tl.arange(0, BLOCK)

    mask = cols < n_cols
    accumulator = 0.0

    a_row_start = row * K

    for k in tl.static_range(0, K):
        a_value = tl.load(a + a_row_start + k)

        b_offsets = k * n_cols + cols
        b_values = tl.load(
            b + b_offsets,
            mask=mask,
            other=0.0,
        )

        accumulator = accumulator + a_value * b_values

    c_offsets = row * n_cols + cols

    tl.store(c + c_offsets, accumulator, mask=mask)
```

The launch grid is:

```python
(m, triton.cdiv(n, BLOCK))
```

The first axis picks the output row. The second axis picks a block of output
columns. Inside one program instance, `cols` is a vector, so each lane computes
one output column for the same row.

The inner loop is over `K`. Because `K` is a `tl.constexpr`, `tl.static_range`
unrolls the loop during tracing. For a tiny `K=3` example, the SSA contains
three copies of the multiply-add pattern:

```text
%4 = load %3, none, none : f32
%13 = load %11, %12, 0.0 : vector<4 x f32>
%14 = mul %4, %13 : vector<4 x f32>
%15 = add 0.0, %14 : vector<4 x f32>

%18 = load %17, none, none : f32
%22 = load %21, %12, 0.0 : vector<4 x f32>
%23 = mul %18, %22 : vector<4 x f32>
%24 = add %15, %23 : vector<4 x f32>

%27 = load %26, none, none : f32
%31 = load %30, %12, 0.0 : vector<4 x f32>
%32 = mul %27, %31 : vector<4 x f32>
%33 = add %24, %32 : vector<4 x f32>
```

There is a useful shape transition here. `a_value` is scalar because every lane
in the block needs the same `A[row, k]`. `b_values` is vector-shaped because
each lane reads a different `B[k, col]`. Multiplying them broadcasts the scalar
across the vector, producing one partial output value per lane.

This kernel is intentionally inefficient. It repeatedly reloads `A` and `B`
from global memory. But it proves that the language can express the outer
structure of matmul:

- a two-dimensional grid;
- vectorized output columns;
- compile-time loop unrolling over `K`;
- scalar-vector broadcasting inside the accumulator.

That is the right stepping stone before introducing tiled shared-memory matmul.

## Long-row sum

The long-row sum kernel uses that compile-time loop:

```python
@triton.jit
def long_row_sum_kernel(x, out, N_COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)

    row_start = row * N_COLS
    partial = 0.0

    for start in tl.static_range(0, N_COLS, BLOCK):
        cols = start + lanes
        mask = cols < N_COLS

        values = tl.load(x + row_start + cols, mask=mask, other=0.0)

        partial = partial + values

    total = tl.sum(partial)
    first_lane = lanes < 1

    tl.store(out + row, total, mask=first_lane)
```

This keeps one CUDA block per row. But instead of each thread loading only one
column, each thread loads a strided sequence:

```text
lane
lane + BLOCK
lane + 2 * BLOCK
...
```

The loop accumulates a partial vector. Each lane sums the columns assigned to
that thread. After the compile-time loop, the kernel performs one ordinary
block-local reduction over the partial vector.

For a small example with `N_COLS=10` and `BLOCK=4`, the loop unrolls into three
loads:

```text
%4 = add 0, %3 : vector<4 x i32>
%7 = load %5, %6, 0.0 : vector<4 x f32>
%8 = add 0.0, %7 : vector<4 x f32>

%10 = add 4, %3 : vector<4 x i32>
%13 = load %11, %12, 0.0 : vector<4 x f32>
%14 = add %8, %13 : vector<4 x f32>

%16 = add 8, %3 : vector<4 x i32>
%19 = load %17, %18, 0.0 : vector<4 x f32>
%20 = add %14, %19 : vector<4 x f32>

%21 = sum %20 : f32
```

There are gaps in the SSA numbering because the optimizer removes some
intermediate operations. That is okay: SSA IDs are stable debug names, not a
promise that the final printed program will be densely numbered.

The important point is that the loop is gone. The compiler sees straight-line
loads, adds, and one final reduction.

This is not a general solution for every reduction. It still uses one block per
row and one final block-local reduction. But it moves the row-width limit from
"at most one block worth of elements" to "as many elements as I am willing to
unroll at compile time."

That is a very Triton-like tradeoff. Compile-time constants can shape the
generated program.

## What changed conceptually

Version 6 is small in the sense that it does not add control flow, a scheduler,
or a layout system.

But it changes the meaning of a vector in the language.

Before this version:

```text
vector value = independent per-thread values
```

After this version:

```text
vector value = independent per-thread values that can also cooperate
```

`tl.sum`, `tl.max`, and `tl.min` are the first operations where a block becomes
more than a bag of unrelated lanes. Shared memory appears in the generated CUDA
because threads need a place to exchange values.

Softmax then composes that new capability with the older elementwise machinery.
The compiler does not learn a new high-level operation called "softmax." It
learns reductions, and the source program builds softmax out of them.

The naive matmul kernel pushes in a different direction. It does not introduce
a new backend primitive, but it combines two program axes, compile-time loop
unrolling, scalar-vector broadcasting, and masked vector stores into a kernel
that looks like the outer shell of real matrix multiplication.

Finally, the naive matmul kernel makes the next missing abstraction very
obvious. Reductions can hide their shared-memory lowering inside one primitive,
but tiled matmul will need source-visible shared state and ordering. That is
the boundary this version walks up to, but does not cross yet.

That is exactly the kind of milestone I like in this project. The language gets
more expressive because the compiler has gained a few precise, composable
ideas rather than one large special case.

## What's next

The next question I want to answer is not about adding another CUDA-specific
kernel. It is about the shape of the compiler itself: how much of mytriton is
really a compiler, and how much of it is just a CUDA string generator?

So the next version takes a detour through
[MLIR](https://mlir.llvm.org/).

The goal is not to replace the CUDA backend yet. Version 6 has many features
that the first MLIR backend will not support. The goal is smaller and more
architectural: take the same optimized SSA for a simple elementwise kernel and
lower it through a second backend.

That means the compiler path becomes:

```text
Python kernel
    -> expression-tree IR
    -> typed SSA
    -> optimization pipeline
    -> CUDA C++ or MLIR GPU dialect
```

The first MLIR backend will be deliberately narrow: one-dimensional elementwise
kernels, `tl.program_id(0)`, `tl.arange(0, BLOCK)`, basic arithmetic, masked
loads, and masked stores. That is much less than the CUDA backend supports, but
it is enough to make the backend boundary real.

Once there are two backends, the compiler has to be more honest about what is
frontend IR, what is backend-specific lowering, and what it means to return
"the generated source" from a launch. That feels like a useful detour before
coming back to tiled matmul.

All code for this milestone is available at
[https://github.com/pbelevich/mytriton/tree/ver6](https://github.com/pbelevich/mytriton/tree/ver6).

Next: [Part 7]({% post_url 2026-06-28-My_Triton_Language_From_Scratch_-_Part_7 %}).
