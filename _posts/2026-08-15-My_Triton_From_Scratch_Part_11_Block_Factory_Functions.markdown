---
layout: post
title:  "My Triton From Scratch Part 11: Block Factory Functions"
date:   2026-08-15 16:00:00 +0000
# categories:
---

In [Part 1: Symbolic Tracing]({% post_url 2026-06-22-My_Triton_From_Scratch_Part_1_Symbolic_Tracing %}),
mytriton learned how to trace Python operators into an expression-tree IR.

In [Part 2: Typed SSA]({% post_url 2026-06-23-My_Triton_From_Scratch_Part_2_Typed_SSA %}),
that tree gained types and an SSA representation.

In [Part 3: CUDA Lowering]({% post_url 2026-06-24-My_Triton_From_Scratch_Part_3_CUDA_Lowering %}),
typed SSA became executable CUDA C++.

In [Part 4: Elementwise Ops]({% post_url 2026-06-25-My_Triton_From_Scratch_Part_4_Elementwise_Ops %}),
the language grew enough elementwise operations for several activation kernels.

In [Part 5: Verification]({% post_url 2026-06-26-My_Triton_From_Scratch_Part_5_Verification %}),
SSA verification and optimization became explicit compiler stages.

In [Part 6: Reductions]({% post_url 2026-06-27-My_Triton_From_Scratch_Part_6_Reductions %}),
threads learned how to cooperate on block-local reductions.

In [Part 7: Minimal MLIR]({% post_url 2026-06-28-My_Triton_From_Scratch_Part_7_Minimal_MLIR %}),
optimized SSA gained a second backend.

In [Part 8: Rank-2 Tiles]({% post_url 2026-06-29-My_Triton_From_Scratch_Part_8_Rank_2_Tiles %}),
blocks gained logical rank-2 shapes and broadcasting.

In [Part 9: AST Frontend]({% post_url 2026-08-01-My_Triton_From_Scratch_Part_9_AST_Frontend %}),
mytriton began interpreting kernel source syntax itself.

In [Part 10: Runtime For Loops]({% post_url 2026-08-08-My_Triton_From_Scratch_Part_10_Runtime_For_Loops %}),
the AST frontend lowered symbolic `range` loops to structured SSA regions and
CUDA `for` loops.

Version 11 fills a much smaller-looking hole: creating a block directly from a
shape, a value, and a dtype.

## The suspicious zero in matmul

The first rank-2 matmul initialized its accumulator like this:

```python
c_offsets = offs_m * N + offs_n
acc = c_offsets * 0.0
```

The expression works because the type rules broadcast the scalar `0.0` across
the `BM x BN` integer offset tile. Numeric promotion changes the element type
to `f32`, producing exactly the desired accumulator type:

```text
block<BM x BN x i32> * f32 -> block<BM x BN x f32>
```

But the code says the wrong thing. The accumulator does not conceptually depend
on output offsets. It needs only two facts:

- its logical shape is `(BM, BN)`;
- its element type is `f32` and every element starts at zero.

The dependency is particularly awkward once runtime loops exist. `acc` is a
loop-carried value, so any expression used to initialize it must be available
before the loop. Pulling address arithmetic into that dependency chain obscures
the real dataflow and makes region ordering harder to read.

What the kernel wants to say is:

```python
acc = tl.zeros((BM, BN), tl.float32)
```

Version 11 adds that operation, along with the more general `tl.full` and
`tl.empty` constructors.

## The public API

The three factory functions are:

```python
tl.empty(shape, dtype)
tl.full(shape, value, dtype)
tl.zeros(shape, dtype)
```

Examples:

```python
acc = tl.zeros((BM, BN), tl.float32)
twos = tl.full([BM, BN], 2.0, tl.float32)
indices = tl.full(BLOCK, runtime_value, tl.int32)
temporary = tl.empty((BLOCK,), tl.float32)
```

The milestone also exposes three dtype objects through `mytriton.language`:

```python
tl.int1
tl.int32
tl.float32
```

They are the public spellings of the compiler's existing scalar types:

```python
int1 = BOOL
int32 = I32
float32 = F32
```

Using dtype objects rather than Python classes keeps construction explicit. A
Python `float` literal describes a scalar value; `tl.float32` describes a
kernel-language element type.

## Normalizing shapes at the language boundary

The API accepts a positive integer or a non-empty list/tuple of positive
integers:

```python
tl.zeros(8, tl.float32)          # shape (8,)
tl.zeros([4, 8], tl.float32)     # shape (4, 8)
tl.zeros((4, 8), tl.float32)     # shape (4, 8)
```

All three forms become one immutable tuple representation immediately:

```python
def _normalize_block_shape(
    shape: int | tuple[int, ...] | list[int],
) -> tuple[int, ...]:
    if type(shape) is int:
        shape = (shape,)
    elif isinstance(shape, list):
        shape = tuple(shape)

    if not isinstance(shape, tuple) or not shape:
        raise TypeError(
            f"block shape must be a non-empty int sequence, got {shape!r}"
        )

    if any(type(dim) is not int or dim <= 0 for dim in shape):
        raise TypeError(
            f"block dimensions must be positive integers, got {shape}"
        )

    return shape
```

Notice the exact `type(dim) is int` check. In Python, `bool` is a subclass of
`int`, but `(4, True)` should not silently mean `(4, 1)` in a compiler type.

Empty and non-positive shapes are rejected as well:

```python
tl.empty((), tl.float32)         # error
tl.zeros((4, 0), tl.float32)     # error
tl.full((4, -1), 0.0, tl.float32)  # error
```

Only the three public scalar dtypes are accepted:

```python
def _require_block_dtype(dtype: ScalarType) -> ScalarType:
    if dtype not in (BOOL, I32, F32):
        raise TypeError(
            f"block dtype must be int1, int32, or float32, got {dtype}"
        )
    return dtype
```

Doing this validation in the language API gives an immediate error for malformed
source. Type inference and SSA verification still repeat the semantic checks,
because compiler IR must remain valid even when it is constructed directly in
a unit test or transformed by a later pass.

## New expression nodes

The factory functions create explicit expression-tree nodes:

```python
@dataclass
class Empty:
    shape: tuple[int, ...]
    dtype: ScalarType


@dataclass
class Full:
    shape: tuple[int, ...]
    value: Expression
    dtype: ScalarType


@dataclass
class Zeros:
    shape: tuple[int, ...]
    dtype: ScalarType
```

`tl.full` unwraps its fill value just like arithmetic and load operations do:

```python
def full(shape, value, dtype) -> Value:
    return Value(
        Full(
            _normalize_block_shape(shape),
            unwrap(value),
            _require_block_dtype(dtype),
        )
    )
```

The value may therefore be a Python scalar or a symbolic runtime scalar:

```python
values = tl.full(BLOCK, 2.5, tl.float32)
values = tl.full(BLOCK, runtime_value, tl.float32)
```

It may not be a block. `tl.full((4,), tl.arange(0, 4), tl.float32)` is ambiguous:
is it a fill, a reshape, or a conversion? Version 11 keeps `full` to its normal
meaning and requires one scalar fill value.

## Type inference

`empty` and `zeros` have a direct type:

```python
elif isinstance(expr, (Empty, Zeros)):
    ty = BlockType(expr.shape, expr.dtype)
```

`full` additionally checks the fill value:

```python
elif isinstance(expr, Full):
    value_ty = self.infer(expr.value)
    if isinstance(value_ty, BlockType):
        raise TypeError(f"full value must be scalar, got {value_ty}")
    self.require_convertible(
        value_ty,
        expr.dtype,
        context="full value",
    )
    ty = BlockType(expr.shape, expr.dtype)
```

Equal types are accepted directly. Numeric `i32` and `f32` may convert in either
direction. Boolean values remain Boolean: Version 11 does not invent implicit
numeric/Boolean conversions.

The result type always comes from the constructor arguments, not from the fill
value. For example:

```python
tl.full((8,), 2, tl.float32)
```

has type `vector<8 x f32>`, even though the Python literal begins as `i32` in
the expression tree.

## Factory operations in SSA

SSA lowering keeps the operation explicit and records normalized shape and
dtype as attributes:

```python
if isinstance(expr, Zeros):
    return self.emit(
        "zeros",
        expr,
        attrs={"shape": expr.shape, "dtype": expr.dtype},
    )
```

For a small kernel:

```python
values = tl.zeros((BLOCK,), tl.float32)
values = values + tl.full([BLOCK], 2.5, tl.float32)
tl.store(out + offsets, values, mask=mask)
```

the relevant SSA is:

```text
%0 = zeros {shape=(8,), dtype=f32} : vector<8 x f32>
%1 = full 2.5 {shape=(8,), dtype=f32} : vector<8 x f32>
%2 = add %0, %1 : vector<8 x f32>
```

`empty` and `zeros` have no operands. Their shape and dtype are compile-time
attributes. `full` has one scalar operand in addition to those attributes:

```text
%1 = full 2.5 {shape=(8,), dtype=f32} : vector<8 x f32>
```

Keeping the constructors in SSA is more useful than immediately rewriting them
to arithmetic. A verifier can check their intent directly, and future backends
can choose a representation appropriate for their execution model.

## Why `zeros` may appear first

After changing the compile-time-unrolled matmul to:

```python
acc = tl.zeros((BM, BN), tl.float32)
```

its SSA begins with:

```text
%0 = zeros {shape=(4, 8), dtype=f32} : block<4x8 x f32>
%1 = program_id {axis=0} : i32
...
```

even though `program_id` and the offsets occur first in Python source.

This is a consequence of lazy, demand-driven lowering. SSA is emitted in
dependency order, not guaranteed source order. The accumulator is needed by
the first unrolled loop update, while some offset expressions are only demanded
by later loads or the final store. `zeros` has no operands, so it can be emitted
as soon as that accumulator dependency is traversed.

The order is valid SSA: every value is defined before use. It also reveals that
the accumulator is now independent of output address calculation, which was the
point of the change.

## Verification

The verifier knows the arity of all three operations:

```python
"empty": 0,
"full": 1,
"zeros": 0,
```

For each constructor it checks that:

- `shape` is a non-empty tuple of positive exact integers;
- `dtype` is one of `bool`, `i32`, or `f32`;
- the declared SSA result type is exactly `BlockType(shape, dtype)`;
- a `full` fill is scalar;
- that scalar is convertible to the requested dtype.

These checks guard against mismatches that normal source construction cannot
produce, such as an operation declaring `shape=(8,)` but returning
`vector<4 x f32>`.

## CUDA lowering is per thread

The current CUDA execution model represents one logical block element with one
scalar value in each CUDA thread. Factory lowering follows that model.

`tl.zeros` assigns a typed zero:

```c++
float v0 = 0.0f;
```

`tl.full` assigns the scalar fill in every participating thread:

```c++
float v1 = 2.5f;
```

For a runtime fill:

```python
values = tl.full(BLOCK, value, tl.float32)
```

each thread gets the same runtime scalar `value`.

`tl.empty` emits an uninitialized local:

```c++
float v0;
```

Reading that value before overwriting it has undefined contents, just as reading
an uninitialized C++ local does.

This is the most important boundary of the milestone:

```text
tl.empty / tl.full / tl.zeros create logical distributed values.
They do not allocate CUDA shared memory.
```

A rank-2 `tl.zeros((BM, BN), tl.float32)` does not create a `BM * BN` array in
each thread either. Under the current one-element-per-thread mapping, it creates
one scalar zero in each thread, and those scalars collectively represent the
logical tile.

## Minimal MLIR lowering

The MLIR backend already scalarizes rank-1 distributed values to one value per
GPU thread. `zeros` and `full` fit that model:

```mlir
%c_f32_0 = arith.constant 0.000000e+00 : f32
%c_f32_2_5 = arith.constant 2.500000e+00 : f32
%v2 = arith.addf %c_f32_0, %c_f32_2_5 : f32
```

When a symbolic numeric fill has a different numeric dtype, lowering emits an
explicit `arith.sitofp` or `arith.fptosi` conversion.

`tl.empty` is intentionally rejected by the MLIR MVP:

```text
MLIR MVP does not support tl.empty
```

An explicit rejection is preferable to inventing an undefined MLIR value or
silently initializing an operation whose semantics say “uninitialized.”

The existing MLIR rank restrictions still apply. Adding constructors does not
make the minimal backend understand rank-2 kernels.

## Matmul becomes honest about its accumulator

The static and runtime matmul kernels can now use:

```python
acc = tl.zeros((BM, BN), tl.float32)

for k in range(K):
    a_values = tl.load(...)
    b_values = tl.load(...)
    acc = acc + a_values * b_values
```

The runtime loop carries that zero tile in exactly the same way as before. Only
the initialization changes:

```text
%10 = zeros {shape=(4, 8), dtype=f32} : block<4x8 x f32>
...
%25 = for %11 in range(0, K, 1) iter_args(%12 = %10) : block<4x8 x f32> {
  ...
  yield %24
}
```

That small source improvement matters to the compiler architecture. Shapes can
now originate from the language directly rather than always being inherited
from `arange`, loads, or address arithmetic.

## Tests

Version 11 covers the new operations at several levels:

- exact SSA and CUDA source for `full + zeros`;
- CUDA execution with integer and floating runtime fill values;
- the uninitialized CUDA declaration for `empty`;
- shape rejection for empty, zero, negative, and Boolean dimensions;
- rejection of block-valued fills;
- MLIR constants and numeric conversion for `full`/`zeros`;
- explicit MLIR rejection of `empty`;
- static and runtime matmul accumulator initialization with `tl.zeros`.

The source-level validation tests are useful, but the exact SSA tests are what
protect the compiler contract. They ensure that shape and dtype survive all the
way from the API to typed IR instead of disappearing into a coincidental scalar
constant.

## What changed conceptually

Before Version 11, a block shape usually came from another expression:

```text
arange/load/address tile
          |
          v
arithmetic used to manufacture the desired type
```

Now a kernel can state the value it wants directly:

```text
(shape, dtype, optional scalar fill)
          |
          v
logical block constructor
          |
          v
typed SSA factory operation
```

This is a language improvement, not yet a memory-system improvement. The new
operations say what logical block value exists. They do not say where a tile is
stored, which threads own which elements, or how threads cooperatively move it
through shared memory.

Those questions require an explicit CUDA layout model. In particular, a future
matmul needs to distinguish the output tile `[BM, BN]`, the `A` tile `[BM, BK]`,
the `B` tile `[BK, BN]`, and the physical threads that cooperate on all three.

All code for this milestone is available at
[https://github.com/pbelevich/mytriton/tree/ver11](https://github.com/pbelevich/mytriton/tree/ver11).

Next: [Part 12: CUDA Tile Layouts]({% post_url 2026-08-22-My_Triton_From_Scratch_Part_12_CUDA_Tile_Layouts %}).
