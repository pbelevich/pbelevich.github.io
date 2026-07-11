---
layout: post
title:  "My Triton From Scratch Part 8: Rank-2 Tiles"
date:   2026-06-29 16:00:00 +0000
# categories:
---

In [Part 1: Symbolic Tracing]({% post_url 2026-06-22-My_Triton_From_Scratch_Part_1_Symbolic_Tracing %}),
mytriton learned how to trace a Python function into an expression-tree IR.

In [Part 2: Typed SSA]({% post_url 2026-06-23-My_Triton_From_Scratch_Part_2_Typed_SSA %}),
it learned how to infer types and lower that tree into typed SSA-style
operations.

In [Part 3: CUDA Lowering]({% post_url 2026-06-24-My_Triton_From_Scratch_Part_3_CUDA_Lowering %}),
it learned how to turn SSA into CUDA C++ and launch the generated kernel.

In [Part 4: Elementwise Ops]({% post_url 2026-06-25-My_Triton_From_Scratch_Part_4_Elementwise_Ops %}),
the language grew enough elementwise operations to write ReLU, leaky ReLU, and
sigmoid.

In [Part 5: Verification]({% post_url 2026-06-26-My_Triton_From_Scratch_Part_5_Verification %}),
the middle of the compiler became stricter: SSA is now verified, optimized, and
verified again before code generation.

In [Part 6: Reductions]({% post_url 2026-06-27-My_Triton_From_Scratch_Part_6_Reductions %}),
vectors learned how to cooperate inside one CUDA block: row-wise sum, max, min,
softmax, and a first naive matmul.

In [Part 7: Minimal MLIR]({% post_url 2026-06-28-My_Triton_From_Scratch_Part_7_Minimal_MLIR %}),
the compiler learned that optimized SSA can feed more than one backend.

Version 8 comes back to CUDA and to matrix-shaped kernels.

The goal is not shared memory yet.

The goal is smaller: make mytriton understand that a block value can have more
than one logical dimension.

Until now, a vector type looked like this:

```text
vector<256 x f32>
```

That was enough for elementwise kernels, reductions, softmax, and the first
naive matmul. But it was awkward for a tile of `C` in matrix multiplication.
A tile is not naturally one row of 256 lanes. It is something like:

```text
block<16x32 x f32>
```

That shape matters. One dimension belongs to rows of `A` and `C`. The other
belongs to columns of `B` and `C`.

Version 8 adds that missing shape.

The code for this milestone is here:
[https://github.com/pbelevich/mytriton/tree/ver8](https://github.com/pbelevich/mytriton/tree/ver8).

## The kernel I want to write

The first target is a matrix add kernel written with a two-dimensional tile:

```python
@triton.jit
def matrix_add_2d_kernel(x, y, out, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)[:, None]
    offs_n = pid_n * BN + tl.arange(0, BN)[None, :]

    offsets = offs_m * N + offs_n
    mask = (offs_m < M) & (offs_n < N)

    lhs = tl.load(x + offsets, mask=mask, other=0.0)
    rhs = tl.load(y + offsets, mask=mask, other=0.0)

    tl.store(out + offsets, lhs + rhs, mask=mask)
```

There are two new bits of syntax here.

The first is:

```python
tl.arange(0, BM)[:, None]
```

and:

```python
tl.arange(0, BN)[None, :]
```

These are the usual Python and NumPy gestures for turning a row or column of
coordinates into something that can broadcast over a matrix-shaped tile.

The second is:

```python
(offs_m < M) & (offs_n < N)
```

Earlier versions used `tl.where`, `tl.maximum`, reductions, and masked loads,
but they did not need a real symbolic boolean `&`. A 2D tile does. The row mask
has shape `BM x 1`. The column mask has shape `1 x BN`. The combined mask has
shape `BM x BN`.

If this works, the same idea can be used for a simple tiled matmul:

```python
@triton.jit
def matmul_2d_kernel(
    a,
    b,
    c,
    M,
    N,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)[:, None]
    offs_n = pid_n * BN + tl.arange(0, BN)[None, :]

    c_offsets = offs_m * N + offs_n
    c_mask = (offs_m < M) & (offs_n < N)

    acc = c_offsets * 0.0

    for k in tl.static_range(0, K):
        a_offsets = offs_m * K + k
        b_offsets = k * N + offs_n

        a_values = tl.load(a + a_offsets, mask=offs_m < M, other=0.0)
        b_values = tl.load(b + b_offsets, mask=offs_n < N, other=0.0)

        acc = acc + a_values * b_values

    tl.store(c + c_offsets, acc, mask=c_mask)
```

This is still a deliberately simple matmul. It does not use shared memory. It
does not tile over `K`. It requires `K` to be a compile-time constant so
`tl.static_range` can unroll the loop.

But it has the source shape I wanted:

```text
one CUDA block computes one BM x BN output tile
```

## VectorType becomes BlockType

The old type system had a `VectorType`:

```python
@dataclass(frozen=True)
class VectorType:
    size: int
    element: ScalarType | PointerType
```

That representation bakes in rank 1. A vector has one size. That was fine when
all distributed values were linear lanes.

Version 8 replaces that with `BlockType`:

```python
@dataclass(frozen=True)
class BlockType:
    shape: tuple[int, ...]
    element: ScalarType | PointerType

    def __post_init__(self):
        if not self.shape:
            raise TypeError("block shape must not be empty")
        if any(dim <= 0 for dim in self.shape):
            raise TypeError(f"block dimensions must be positive, got {self.shape}")

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def num_elements(self) -> int:
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    @property
    def size(self) -> int:
        if self.rank != 1:
            raise TypeError(f"rank-{self.rank} block has no single size")
        return self.shape[0]
```

The old public constructor still exists as a compatibility helper:

```python
def VectorType(size: int, element: ScalarType | PointerType) -> BlockType:
    return BlockType((size,), element)
```

That lets most of the older tests keep saying `VectorType(256, F32)`. They now
get a rank-1 `BlockType` under the hood.

The printer keeps the old spelling for rank-1 blocks:

```python
def __str__(self):
    if self.rank == 1:
        return f"vector<{self.shape[0]} x {self.element}>"

    shape = "x".join(str(dim) for dim in self.shape)
    return f"block<{shape} x {self.element}>"
```

That is why existing SSA still prints as:

```text
vector<256 x f32>
```

while new rank-2 values print as:

```text
block<16x32 x f32>
```

This is a small compatibility trick, but it makes the transition much easier to
read. Rank-1 code still looks like rank-1 code. Only the new tile-shaped values
use the new spelling.

## Broadcasting is now shape broadcasting

The old type inference had a simple rule:

```text
scalar + vector<N> -> vector<N>
vector<N> + vector<N> -> vector<N>
vector<N> + vector<M> -> error, unless N == M
```

That is not enough for:

```python
offs_m * N + offs_n
```

because `offs_m * N` has shape `BM x 1`, while `offs_n` has shape `1 x BN`.
Those should combine into `BM x BN`.

So shape handling moves into a small helper:

```python
def broadcast_shapes(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    if not shapes:
        return ()

    max_rank = max(len(shape) for shape in shapes)
    padded = [(1,) * (max_rank - len(shape)) + shape for shape in shapes]

    dims = []
    for dim_values in zip(*padded, strict=True):
        non_ones = {dim for dim in dim_values if dim != 1}
        if len(non_ones) > 1:
            rendered = ", ".join(
                "x".join(str(dim) for dim in shape) for shape in shapes
            )
            raise ValueError(f"cannot broadcast shapes: {rendered}")

        dims.append(next(iter(non_ones), 1))

    return tuple(dims)
```

Then type inference can ask for a result with the right element type and the
broadcasted shape:

```python
def with_shape(
    self,
    element: ScalarType | PointerType,
    *types: Type,
) -> Type:
    shapes = [ty.shape for ty in types if isinstance(ty, BlockType)]
    if not shapes:
        return element
    try:
        shape = broadcast_shapes(*shapes)
    except ValueError as error:
        raise TypeError(f"Cannot broadcast shapes: {shapes}") from error

    return BlockType(shape, element)
```

This same helper is used by arithmetic, comparisons, pointer arithmetic, loads,
stores, masks, and `tl.where`.

That is the important part. Rank-2 tiles are not a special case in every
operation. Most operations only need to say:

```text
my element type is X
my shape is the broadcast of my operands
```

For example, pointer addition keeps its old element rule but gets rank-2 shapes
for free:

```python
elif isinstance(expr, AddPtr):
    base = self.infer(expr.base)
    offset = self.infer(expr.offset)
    base_element = self.element_type(base)

    if not isinstance(base_element, PointerType):
        raise TypeError(f"Expected pointer, got {base}")

    if self.element_type(offset) != I32:
        raise TypeError(f"Pointer offset must be i32, got {offset}")

    ty = self.with_shape(base_element, base, offset)
```

If `offset` is `block<16x32 x i32>`, then `out + offset` becomes:

```text
block<16x32 x ptr<f32>>
```

That is exactly what a tile store needs.

## Expanding dimensions

The frontend only needs a tiny slice of indexing syntax:

```python
x[:, None]
x[None, :]
```

That is implemented directly on symbolic values:

```python
def __getitem__(self, index) -> Value:
    if index == (slice(None), None):
        return Value(ExpandDims(self.expr, axis=1))

    if index == (None, slice(None)):
        return Value(ExpandDims(self.expr, axis=0))

    raise TypeError(
        "only x[:, None] and x[None, :] are supported for symbolic values"
    )
```

There is no general symbolic indexing yet. This is intentionally narrow.

The expression-tree node is also small:

```python
@dataclass
class ExpandDims:
    value: Any
    axis: int
```

Type inference inserts a size-1 dimension:

```python
elif isinstance(expr, ExpandDims):
    value_ty = self.infer(expr.value)
    if not isinstance(value_ty, BlockType):
        raise TypeError(f"expand_dims expects block, got {value_ty}")
    axis = expr.axis
    if axis < 0 or axis > value_ty.rank:
        raise TypeError(f"invalid expand_dims axis {axis} for {value_ty}")
    shape = (*value_ty.shape[:axis], 1, *value_ty.shape[axis:])
    ty = BlockType(shape, value_ty.element)
```

So:

```text
vector<16 x i32>
```

becomes:

```text
block<16x1 x i32>
```

and:

```text
vector<32 x i32>
```

becomes:

```text
block<1x32 x i32>
```

The SSA lowering just preserves that operation:

```text
%2 = arange {start=0, end=16} : vector<16 x i32>
%3 = expand_dims %2 {axis=1} : block<16x1 x i32>
%8 = arange {start=0, end=32} : vector<32 x i32>
%9 = expand_dims %8 {axis=0} : block<1x32 x i32>
```

At this point, the middle-end understands the shape. The backend still needs to
decide how that shape maps to CUDA threads.

## Boolean masks need `&`

The mask for a 2D tile is naturally a conjunction:

```python
mask = (offs_m < M) & (offs_n < N)
```

Symbolic Python values now implement `&`:

```python
def __and__(self, other: Value | bool) -> Value:
    return Value(BinOp("&", self.expr, unwrap(other)))

def __rand__(self, other: Value | bool) -> Value:
    return Value(BinOp("&", unwrap(other), self.expr))
```

Type inference checks that both operands are Boolean blocks or scalars:

```python
elif expr.op == "&":
    if self.element_type(lhs) != BOOL or self.element_type(rhs) != BOOL:
        raise TypeError(f"& expects bool operands, got {lhs} and {rhs}")
    ty = self.with_shape(BOOL, lhs, rhs)
```

The parentheses in the source are not decoration. In Python, comparisons and
bitwise operators have their own precedence rules, and:

```python
offs_m < M & offs_n < N
```

does not mean what a Triton programmer wants it to mean.

mytriton cannot change Python's parser, but it can make the error friendlier.
The symbolic boolean conversion now says:

```python
def __bool__(self) -> bool:
    raise TypeError(
        "Python control flow over symbolic values is not supported; "
        "when combining comparisons with &, wrap each comparison in parentheses"
    )
```

That is a small usability improvement, but it matters. The first time I wrote a
2D mask, I got the parentheses wrong.

## Choosing the CUDA block shape

Earlier versions selected the CUDA thread block size by scanning rank-1 vector
widths. If all distributed values had width 256, the kernel launched 256 CUDA
threads per block.

Rank-2 tiles need a slightly different question:

```text
what is the logical block shape?
```

The helper starts by collecting all SSA result block shapes:

```python
def result_block_shapes(ssa_ops: list[SSAOp]) -> list[tuple[int, ...]]:
    return [
        op.result.ty.shape
        for op in ssa_ops
        if op.result is not None and isinstance(op.result.ty, BlockType)
    ]
```

Then CUDA lowering chooses a rank-1 or rank-2 kernel shape:

```python
def cuda_kernel_block_shape(ssa_ops: list[SSAOp]) -> tuple[int, ...]:
    shapes = result_block_shapes(ssa_ops)

    if not shapes:
        return (1,)

    if any(len(shape) > 2 for shape in shapes):
        rendered = ", ".join(str(shape) for shape in shapes)
        raise ValueError(
            f"CUDA lowering supports only rank-1/rank-2 blocks, got {rendered}"
        )
```

For rank-1 kernels, the old rule still applies:

```python
widths = {shape[0] for shape in rank1_shapes}
if len(widths) != 1:
    rendered = ", ".join(str(width) for width in sorted(widths))
    raise ValueError(f"CUDA lowering requires one vector width, got: {rendered}")

return (next(iter(widths)),)
```

For rank-2 kernels, the rank-2 shapes are broadcast together:

```python
if rank2_shapes:
    block_shape = broadcast_shapes(*rank2_shapes)

    if len(block_shape) != 2:
        raise ValueError(f"expected rank-2 CUDA block shape, got {block_shape}")
```

There is one subtlety. In a rank-2 kernel, rank-1 aranges appear in the SSA:

```text
arange(0, BM) -> vector<BM>
arange(0, BN) -> vector<BN>
```

Those are not independent execution widths. They are coordinate vectors that
will become the two tile axes.

So rank-1 widths are allowed if they match one tile dimension or the full tile
size:

```python
allowed_rank1_widths = {block_shape[0], block_shape[1], prod(block_shape)}

bad_widths = [
    shape[0] for shape in rank1_shapes if shape[0] not in allowed_rank1_widths
]

if bad_widths:
    raise ValueError(
        "rank-1 block widths in a rank-2 CUDA kernel must match "
        f"one tile dimension or full tile size; got {bad_widths}, "
        f"tile shape is {block_shape}"
    )
```

Finally, the CUDA thread count is the product of the chosen shape:

```python
def cuda_threads_per_block(ssa_ops: list[SSAOp]) -> int:
    block_shape = cuda_kernel_block_shape(ssa_ops)
    threads_per_block = prod(block_shape)

    if not 1 <= threads_per_block <= 1024:
        raise ValueError(
            "CUDA threads per block must be between 1 and 1024, "
            f"got {threads_per_block}"
        )

    return threads_per_block
```

For a `16 x 32` tile, that means 512 CUDA threads.

The logical shape is 2D. The CUDA thread block is still launched as a flat
one-dimensional block.

## Mapping a flat thread block to a tile

CUDA lowering starts by asking for the block shape:

```python
self.block_shape = cuda_kernel_block_shape(ssa_ops)
```

If the shape is rank 2, the backend emits two local coordinates:

```python
def emit_rank2_prologue(self) -> None:
    if not self.is_rank2_kernel():
        return

    _, cols = self.block_shape

    self.lines.extend(
        [
            f"    int tile_i = threadIdx.x / {cols};",
            f"    int tile_j = threadIdx.x % {cols};",
        ]
    )
```

For a `4 x 8` tile, the generated CUDA begins:

```cuda
extern "C" __global__
void matmul_2d_kernel(float* a, float* b, float* c, int M, int N) {
    int tile_i = threadIdx.x / 8;
    int tile_j = threadIdx.x % 8;
    ...
}
```

Each CUDA thread still holds one scalar value for each SSA block result. The
difference is that the backend can now decide whether that scalar belongs to
tile row `tile_i`, tile column `tile_j`, or their broadcasted combination.

## Delaying arange lowering

In a rank-1 kernel, `tl.arange(0, BLOCK)` lowers directly to `threadIdx.x`:

```cuda
int v2 = threadIdx.x;
```

In a rank-2 kernel, `arange(0, BM)` and `arange(0, BN)` need to become different
coordinates after `expand_dims`.

So the CUDA backend stores a tiny internal reference instead of immediately
emitting a local:

```python
@dataclass(frozen=True)
class CudaArangeRef:
    start: int
    end: int

    @property
    def width(self) -> int:
        return self.end - self.start
```

The `arange` lowering becomes:

```python
elif op.opcode == "arange":
    start = op.attrs["start"]
    end = op.attrs["end"]

    if self.is_rank2_kernel():
        self.values[result.id] = CudaArangeRef(start=start, end=end)
    else:
        expression = "threadIdx.x" if start == 0 else f"({start} + threadIdx.x)"
        self.assign(result, expression)
```

Then `expand_dims` consumes that reference:

```python
elif op.opcode == "expand_dims":
    if not self.is_rank2_kernel():
        raise TypeError(
            "CUDA expand_dims lowering currently requires rank-2 kernel"
        )

    arange_ref = self.operand(operand)
    if not isinstance(arange_ref, CudaArangeRef):
        raise TypeError(
            "CUDA expand_dims MVP supports only direct arange expansion, "
            f"got {arange_ref}"
        )

    axis = op.attrs["axis"]
    rows, cols = self.block_shape
```

The shape of the result selects the tile coordinate:

```python
if axis == 1 and result_shape == (rows, 1):
    coord = "tile_i"
elif axis == 0 and result_shape == (1, cols):
    coord = "tile_j"
else:
    raise TypeError(
        f"cannot map expand_dims result {result.ty} into CUDA tile "
        f"shape {self.block_shape}"
    )
```

For the matrix add kernel, that gives:

```cuda
int v3 = tile_i;
int v9 = tile_j;
```

After that, normal scalar CUDA expressions take over:

```cuda
int v4 = (v1 + v3);
int v10 = (v7 + v9);
int v11 = (v5 + v10);
```

The rank-2 shape is gone from the local CUDA variable declarations, just like
rank-1 vector types disappeared in earlier versions. The shape still controlled
which thread coordinate each value uses.

## The SSA for a tile

The matrix add SSA shows the new shape story without the noise of matmul:

```text
%0 = program_id {axis=0} : i32
%1 = mul %0, 16 : i32
%2 = arange {start=0, end=16} : vector<16 x i32>
%3 = expand_dims %2 {axis=1} : block<16x1 x i32>
%4 = add %1, %3 : block<16x1 x i32>
%5 = mul %4, N : block<16x1 x i32>
%6 = program_id {axis=1} : i32
%7 = mul %6, 32 : i32
%8 = arange {start=0, end=32} : vector<32 x i32>
%9 = expand_dims %8 {axis=0} : block<1x32 x i32>
%10 = add %7, %9 : block<1x32 x i32>
%11 = add %5, %10 : block<16x32 x i32>
%12 = addptr x, %11 : block<16x32 x ptr<f32>>
%13 = cmp_lt %4, M : block<16x1 x bool>
%14 = cmp_lt %10, N : block<1x32 x bool>
%15 = and %13, %14 : block<16x32 x bool>
%16 = load %12, %15, 0.0 : block<16x32 x f32>
```

This is the part I wanted the compiler to make visible.

The row coordinate has shape `16 x 1`.

The column coordinate has shape `1 x 32`.

The linear offset has shape `16 x 32`.

The mask has shape `16 x 32`.

The load has shape `16 x 32`.

Nothing in the SSA says "this is matrix add" or "this is matrix multiplication."
It only says that values have shapes and operations broadcast over those shapes.

That makes the matmul SSA almost boring in the right way:

```text
%17 = load %15, %16, 0.0 : block<4x1 x f32>
%22 = load %20, %21, 0.0 : block<1x8 x f32>
%23 = mul %17, %22 : block<4x8 x f32>
%24 = add %12, %23 : block<4x8 x f32>
```

The `A` load is a column-shaped block.

The `B` load is a row-shaped block.

Multiplication broadcasts them into the output tile.

That is the same mathematical shape as an outer product. The compiler does not
need to have an `outer_product` operation to represent it.

## The generated CUDA for matmul

For the test case with `BM = 4`, `BN = 8`, and `K = 3`, the generated CUDA
starts like this:

```cuda
extern "C" __global__
void matmul_2d_kernel(float* a, float* b, float* c, int M, int N) {
    int tile_i = threadIdx.x / 8;
    int tile_j = threadIdx.x % 8;
    int v0 = blockIdx.x;
    int v1 = (v0 * 4);
    int v3 = tile_i;
    int v4 = (v1 + v3);
    int v5 = (v4 * N);
    int v6 = blockIdx.y;
    int v7 = (v6 * 8);
    int v9 = tile_j;
    int v10 = (v7 + v9);
    int v11 = (v5 + v10);
    float v12 = (v11 * 0.0f);
    ...
}
```

The accumulator initialization is a little silly:

```python
acc = c_offsets * 0.0
```

That creates a symbolic zero tile with the right shape. mytriton does not have
a `tl.full` or `tl.zeros` operation yet, so this is a convenient way to ask the
existing type system for:

```text
block<BM x BN x f32>
```

Then the unrolled loop emits one outer-product-shaped update per `k`:

```cuda
float v17 = (v16 ? a[v13] : 0.0f);
float v22 = (v21 ? b[v19] : 0.0f);
float v23 = (v17 * v22);
float v24 = (v12 + v23);

float v29 = (v16 ? a[v26] : 0.0f);
float v34 = (v21 ? b[v31] : 0.0f);
float v35 = (v29 * v34);
float v36 = (v24 + v35);

float v41 = (v16 ? a[v38] : 0.0f);
float v46 = (v21 ? b[v43] : 0.0f);
float v47 = (v41 * v46);
float v48 = (v36 + v47);
```

Finally, the store uses the combined rank-2 mask:

```cuda
bool v52 = (v16 && v21);
if (v52) {
    c[v11] = v48;
}
```

Each CUDA thread computes one output element. The tile shape controls which
`A` element and which `B` element that thread reads at each unrolled `k`.

Again, this is not fast matmul. It is the smallest matmul that proves the
compiler can represent a tile.

## MLIR stays narrow

Version 7 introduced the MLIR backend, but Version 8 does not extend it to
rank-2 tiles.

That is intentional. The MLIR backend is still the honest MVP from the previous
part: one-dimensional elementwise kernels only.

It now explicitly rejects rank-2 block results before trying to emit MLIR:

```python
for op in ssa_ops:
    if op.result is None:
        continue
    ty = op.result.ty
    if isinstance(ty, BlockType) and ty.rank != 1:
        raise TypeError(f"MLIR MVP supports only rank-1 blocks, got {ty}")
```

This follows the same rule as Part 7. A limited backend is fine. A backend that
silently lowers `block<16x32 x f32>` as if it were `vector<512 x f32>` is not.

The CUDA backend gets the new feature first because the whole point of this
version is to understand the CUDA execution mapping for rank-2 tiles.

MLIR can catch up later.

## Tests

Most of the older tests did not need to change because `VectorType` is still a
rank-1 constructor. That is a good sign. The existing elementwise kernels,
reductions, softmax, and naive matmul continue to exercise the old shape path.

The new tests focus on three things.

First, rank-2 SSA lowering:

```python
def tile_shape_kernel(out, M, N, BM: tl.constexpr, BN: tl.constexpr):
    offs_m = tl.arange(0, BM)[:, None]
    offs_n = tl.arange(0, BN)[None, :]

    offsets = offs_m * N + offs_n
    mask = (offs_m < M) & (offs_n < N)

    tl.store(out + offsets, offsets, mask=mask)
```

The expected SSA checks that the shapes are exactly what I think they are:

```text
%1 = expand_dims %0 {axis=1} : block<16x1 x i32>
%4 = expand_dims %3 {axis=0} : block<1x32 x i32>
%5 = add %2, %4 : block<16x32 x i32>
%9 = and %7, %8 : block<16x32 x bool>
store %6, %5, %9
```

Second, CUDA source generation for rank-2 matrix add:

```python
assert "int tile_i = threadIdx.x / 32;" in cuda_src
assert "int tile_j = threadIdx.x % 32;" in cuda_src
```

The source test also checks the full generated CUDA for a representative tile.
That is intentionally brittle. When a compiler is this small, exact source
tests are useful because they make lowering choices visible.

Third, rank-2 matmul execution when CUDA is available:

```python
expected = a @ b
cp.testing.assert_allclose(c, expected, rtol=1e-5, atol=1e-5)
```

On machines without a CUDA GPU, execution tests are skipped. Codegen and type
tests still run normally.

## Current boundaries

Version 8 is still quite small.

Only rank-1 and rank-2 block values are supported by CUDA lowering. Rank-3
blocks are rejected.

`expand_dims` lowering is intentionally narrow. It supports direct expansion of
an `arange` into the row or column coordinate of the current tile. It does not
yet support arbitrary reshapes or expanding a computed block value.

The CUDA block is still launched as one-dimensional. A rank-2 tile is mapped to
that flat thread block with division and modulo. That keeps launch logic simple,
but it is only one possible mapping.

Reductions are still rank-1. There is no reduction over one dimension of a
rank-2 tile yet.

Matrix multiplication still has no source-visible shared memory, no barriers,
and no `K` tiling. It repeatedly reads from global memory and relies on
`tl.static_range` to unroll a constexpr `K`.

Those boundaries are real, but they are useful boundaries. Version 8 is not
trying to be a good GEMM implementation. It is trying to give the compiler a
language for tile shapes.

## What changed conceptually

Before Version 8, a distributed value in mytriton meant:

```text
one logical vector lane per CUDA thread
```

That model was simple and surprisingly productive. It carried the project
through elementwise kernels, reductions, softmax, and a first matmul.

But it flattened too much information.

After Version 8, a distributed value means:

```text
one logical block element per CUDA thread
```

For rank-1 blocks, that is the same as before.

For rank-2 blocks, the compiler now knows that the value has rows and columns.
The backend can still map those rows and columns onto a flat `threadIdx.x`, but
the IR no longer has to pretend the tile was always a vector.

That is the important shift:

```text
execution can be flat
IR shape does not have to be
```

The matmul kernel benefits immediately. Loading `A` produces a `BM x 1` block.
Loading `B` produces a `1 x BN` block. Multiplying them produces a `BM x BN`
block. That is the shape of the computation, and now the SSA says so.

This also makes the next missing pieces easier to name.

A real tiled matmul needs a way to express shared memory tiles, synchronization,
and probably a less toy layout story. But those features should sit on top of
block-shaped values, not replace them.

So Version 8 does not cross into shared memory yet. It builds the floor under
that step.

All code for this milestone is available at
[https://github.com/pbelevich/mytriton/tree/ver8](https://github.com/pbelevich/mytriton/tree/ver8).
