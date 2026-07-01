---
layout: post
title:  "My Triton From Scratch Part 4: Elementwise Ops"
date:   2026-06-25 13:00:00 +0000
# categories:
---

In [Part 1: Symbolic Tracing]({% post_url 2026-06-22-My_Triton_From_Scratch_Part_1_Symbolic_Tracing %}),
mytriton learned how to trace a Triton-looking Python function into an
expression tree.

In [Part 2: Typed SSA]({% post_url 2026-06-23-My_Triton_From_Scratch_Part_2_Typed_SSA %}),
it learned how to turn that tree into a typed SSA-style IR.

In [Part 3: CUDA Lowering]({% post_url 2026-06-24-My_Triton_From_Scratch_Part_3_CUDA_Lowering %}),
it learned how to lower that SSA into readable CUDA C++ and, when the inputs
are CuPy arrays, actually launch the generated kernel.

That was enough to run vector addition on a GPU. But vector addition is a very
patient kernel. It only asks for a few binary operators, a comparison, pointer
arithmetic, masked loads, and masked stores.

Version 4 makes the language feel less like it was custom-built for one demo.
It adds a small set of elementwise operations:

- unary negation;
- [`tl.exp`](https://triton-lang.org/main/python-api/generated/triton.language.exp.html);
- [`tl.minimum`](https://triton-lang.org/main/python-api/generated/triton.language.minimum.html);
- [`tl.maximum`](https://triton-lang.org/main/python-api/generated/triton.language.maximum.html);
- `tl.where`.

Those operations are deliberately modest. There is no new execution system and
no new layout machinery. The interesting part is that each operation has to pass
through every compiler layer that already exists:

```text
Python syntax
    -> expression-tree node
    -> type inference
    -> SSA opcode
    -> CUDA expression
```

That is the whole point of this milestone. A small language feature is only
really added when all layers agree on what it means.

The code for this milestone is here:
[https://github.com/pbelevich/mytriton/tree/ver4](https://github.com/pbelevich/mytriton/tree/ver4).

## The kernels I want to write

The new operations are enough to write a ReLU kernel:

```python
@triton.jit
def relu_kernel(x, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n

    a = tl.load(x + offs, mask=mask, other=0.0)

    tl.store(out + offs, tl.maximum(a, 0.0), mask=mask)
```

They are also enough to write leaky ReLU:

```python
@triton.jit
def leaky_relu_kernel(x, out, n, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n

    a = tl.load(x + offs, mask=mask, other=0.0)

    b = tl.where(a < 0.0, alpha * a, a)

    tl.store(out + offs, b, mask=mask)
```

and a sigmoid kernel:

```python
@triton.jit
def sigmoid_kernel(x, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n

    a = tl.load(x + offs, mask=mask, other=0.0)

    b = 1.0 / (1.0 + tl.exp(-a))

    tl.store(out + offs, b, mask=mask)
```

I like these examples because they stress the current compiler in slightly
different ways.

ReLU introduces a named elementwise operation, `tl.maximum`, rather than another
Python operator overload.

Leaky ReLU introduces value selection. Its condition is a vector mask, but that
mask no longer guards a memory operation. It chooses between two computed
values.

Sigmoid composes several expression forms: a masked load, unary negation,
`tl.exp`, scalar-vector addition, scalar-vector division, and a masked store.
There is no special sigmoid node. If the compiler can lower the pieces, the
larger expression falls out naturally.

## Unary operations

Until now, most value-producing operations were binary. A `Value` could be
added, multiplied, compared, divided, or used in pointer arithmetic. All of
those operations had two operands.

Negation is the first true unary operator in the language:

```python
-a
```

The frontend records it with a new node:

```python
@dataclass
class UnaryOp:
    op: str
    value: Any
```

and `Value.__neg__` creates one:

```python
def __neg__(self) -> Value:
    return Value(UnaryOp("neg", self.expr))
```

I could have made a separate `Neg` dataclass. For now, `UnaryOp("neg", value)`
is more useful because `tl.exp` has the same shape:

```python
def exp(value: Value | float) -> Value:
    return Value(UnaryOp("exp", unwrap(value)))
```

Both operations have one input and one output. They differ in the operation name
and in their type rules.

This is a small design choice, but it is a nice example of the kind of pressure
that gradually shapes an IR. If the language grows many unary operations, a
shared `UnaryOp` node keeps the frontend compact. If some unary operation later
needs special attributes or unusual semantics, it can still become its own node.

For Version 4, the shared node is enough.

## Named extrema

Minimum and maximum are also elementwise operations, but they are not Python
operators in this little language. They enter through the public `tl` module:

```python
def maximum(lhs: Value | int | float, rhs: Value | int | float) -> Value:
    return Value(Maximum(unwrap(lhs), unwrap(rhs)))


def minimum(lhs: Value | int | float, rhs: Value | int | float) -> Value:
    return Value(Minimum(unwrap(lhs), unwrap(rhs)))
```

Their expression nodes are plain binary nodes with more specific names:

```python
@dataclass
class Maximum:
    lhs: Any
    rhs: Any


@dataclass
class Minimum:
    lhs: Any
    rhs: Any
```

I kept them separate from `BinOp` on purpose.

`BinOp` currently represents syntax like `+`, `*`, `/`, and `<`. Those are
Python operators that my symbolic `Value` intercepts. `tl.maximum` and
`tl.minimum` are library-style language functions. Keeping them as explicit
nodes makes the expression tree say what the source meant, instead of pretending
everything is an overloaded Python operator.

That distinction also makes later lowering a little clearer. `Maximum` becomes
the SSA opcode `maximum`. `Minimum` becomes `minimum`. There is no need to
encode them as a generic call.

## Where becomes select

`tl.where` is different from minimum, maximum, and `exp`. It does not compute a
new numeric value directly. It chooses between two values using a Boolean
condition:

```python
tl.where(condition, true_value, false_value)
```

In leaky ReLU, the condition is itself an expression:

```python
tl.where(a < 0.0, alpha * a, a)
```

The frontend node is:

```python
@dataclass
class Where:
    condition: Any
    true_value: Any
    false_value: Any
```

and the public function is only a thin wrapper:

```python
def where(
    condition: Value | bool,
    true_value: Value | int | float,
    false_value: Value | int | float,
) -> Value:
    return Value(Where(unwrap(condition), unwrap(true_value), unwrap(false_value)))
```

The name `where` belongs to the source language. The SSA operation is called
`select`:

```text
%9 = select %7, %8, %6 : vector<256 x f32>
```

That name is common in compiler IRs: a select operation takes a condition, a
true value, and a false value, then produces one result. There is no control
flow edge and no branch. It is still a value-producing expression.

The value arms are ordinary SSA values, not lazy Python branches. In leaky ReLU,
`alpha * a` is computed as its own SSA value before `select` chooses between it
and `a`.

That distinction matters for mytriton. Before this point, masks only guarded
memory operations:

```python
tl.load(..., mask=mask, other=0.0)
tl.store(..., mask=mask)
```

With `tl.where`, a Boolean vector can participate in ordinary value computation.
It stops being only a memory safety guard and becomes data.

## Type inference for the new operations

The type system from
[Part 2: Typed SSA]({% post_url 2026-06-23-My_Triton_From_Scratch_Part_2_Typed_SSA %})
already knows the two ideas these operations need:

- numeric promotion;
- scalar-vector broadcasting.

Minimum and maximum use the same numeric promotion rule as arithmetic:

```python
elif isinstance(expr, (Maximum, Minimum)):
    lhs = self.infer(expr.lhs)
    rhs = self.infer(expr.rhs)
    element = self.promote(lhs, rhs)
    ty = self.with_shape(element, lhs, rhs)
```

If either side is `f32`, the result is `f32`. Otherwise, two integer operands
produce `i32`. If one operand is a vector and the other is a scalar, the scalar
is broadcast across the vector shape:

```text
maximum(vector<256 x f32>, f32) -> vector<256 x f32>
minimum(i32, f32)              -> f32
```

Boolean extrema are rejected. That is not because it would be impossible to
define them, but because I do not want Boolean values quietly drifting through
numeric operations.

Unary operations have their own checks:

```python
elif isinstance(expr, UnaryOp):
    value = self.infer(expr.value)
    unary_element = self.element_type(value)
    if expr.op == "neg":
        if unary_element not in (I32, F32):
            raise TypeError(f"Cannot negate {value}")
    elif expr.op == "exp":
        if unary_element != F32:
            raise TypeError(f"exp requires f32, got {value}")
    else:
        raise TypeError(f"Unknown unary operation: {expr.op}")
    ty = value
```

Negation accepts integers and floats. `tl.exp` only accepts `f32`.

The result keeps the input shape. A scalar `f32` remains `f32`; a
`vector<256 x f32>` remains `vector<256 x f32>`.

This shape preservation is what lets sigmoid work without any special casing.
The loaded value `a` is a vector. `-a` is a vector. `tl.exp(-a)` is a vector.
`1.0 + tl.exp(-a)` broadcasts the scalar `1.0` over that vector. The final
division does the same thing.

`Where` combines the rules for masks and branch values:

```python
elif isinstance(expr, Where):
    condition = self.infer(expr.condition)
    if self.element_type(condition) != BOOL:
        raise TypeError(f"where condition must be bool, got {condition}")
    true_ty = self.infer(expr.true_value)
    false_ty = self.infer(expr.false_value)
    element = self.promote(true_ty, false_ty)
    ty = self.with_shape(element, condition, true_ty, false_ty)
```

The condition must be Boolean. It also participates in shape inference, so a
vector condition can broadcast scalar value arms:

```text
where(vector<256 x bool>, f32, f32) -> vector<256 x f32>
```

The two value arms use the same numeric promotion rules as arithmetic. In the
leaky-ReLU kernel, all three pieces are vectors with the same lane count:

```text
a < 0.0       : vector<256 x bool>
alpha * a     : vector<256 x f32>
a             : vector<256 x f32>
where(...)    : vector<256 x f32>
```

## Lowering to SSA

Once the expression tree has new node types, SSA lowering needs to emit new
opcodes.

Minimum and maximum lower like ordinary binary operations:

```python
if isinstance(expr, (Maximum, Minimum)):
    lhs = self.lower_expr(expr.lhs)
    rhs = self.lower_expr(expr.rhs)

    return self.emit(
        "maximum" if isinstance(expr, Maximum) else "minimum",
        expr,
        operands=(lhs, rhs),
    )
```

Unary operations lower after their single operand:

```python
if isinstance(expr, UnaryOp):
    value = self.lower_expr(expr.value)

    return self.emit(
        expr.op,
        expr,
        operands=(value,),
    )
```

`Where` lowers after its condition and both value arms:

```python
if isinstance(expr, Where):
    condition = self.lower_expr(expr.condition)
    true_value = self.lower_expr(expr.true_value)
    false_value = self.lower_expr(expr.false_value)
    return self.emit(
        "select",
        expr,
        operands=(
            condition,
            true_value,
            false_value,
        ),
    )
```

That means the SSA printer now has a few new opcodes to show, but no new
structural concept. There are still no basic blocks and no control flow in the
IR. These are still straight-line value definitions.

For ReLU, the interesting part of the SSA is only one line:

```text
%6 = load %4, %5, 0.0 : vector<256 x f32>
%7 = maximum %6, 0.0 : vector<256 x f32>
store %8, %7, %5
```

The source expression:

```python
tl.maximum(a, 0.0)
```

has become an explicit typed operation:

```text
maximum %6, 0.0 : vector<256 x f32>
```

For leaky ReLU, the new operation is `select`:

```text
%6 = load %4, %5, 0.0 : vector<256 x f32>
%7 = cmp_lt %6, 0.0 : vector<256 x bool>
%8 = mul alpha, %6 : vector<256 x f32>
%9 = select %7, %8, %6 : vector<256 x f32>
store %10, %9, %5
```

The condition `%7` chooses between the scaled negative path `%8` and the
original value `%6`.

For sigmoid, the new operations compose with the old arithmetic opcodes:

```text
%6 = load %4, %5, 0.0 : vector<256 x f32>
%7 = neg %6 : vector<256 x f32>
%8 = exp %7 : vector<256 x f32>
%9 = add 1.0, %8 : vector<256 x f32>
%10 = div 1.0, %9 : vector<256 x f32>
store %11, %10, %5
```

This is the part I find most satisfying. There is no "sigmoid lowering" in the
compiler. Sigmoid is just a Python expression that happens to build a useful
composition of smaller compiler operations.

That is the difference between adding a special case and growing a language.

## Lowering unary operations and select to CUDA

[Part 3: CUDA Lowering]({% post_url 2026-06-24-My_Triton_From_Scratch_Part_3_CUDA_Lowering %})
established the main CUDA lowering rule:

```text
one SSA vector lane == one CUDA thread-local scalar value
```

The new unary operations fit directly into that rule.

Negation lowers to a scalar CUDA expression:

```python
elif op.opcode == "neg":
    value = self.expression_operand(op.operands[0])
    self.assign(result, f"-({value})")
```

So this SSA:

```text
%7 = neg %6 : vector<256 x f32>
```

becomes:

```cuda
float v7 = -(v6);
```

`tl.exp` lowers to CUDA's `expf`:

```python
elif op.opcode == "exp":
    value = self.expression_operand(op.operands[0])
    if self.scalar_type(result.ty) != F32:
        raise TypeError(f"exp requires f32, got {result.ty}")
    self.assign(result, f"expf({value})")
```

and in the sigmoid kernel:

```cuda
float v8 = expf(v7);
```

The type inference layer should already have rejected non-`f32` inputs to
`exp`, but the CUDA backend checks again. I like that redundancy in a small
compiler. It means that if a malformed SSA operation reaches code generation,
the backend fails near the place where the illegal target operation would be
printed.

`select` lowers to CUDA's conditional operator:

```python
elif op.opcode == "select":
    condition = self.expression_operand(op.operands[0])
    true_value = self.expression_operand(op.operands[1])
    false_value = self.expression_operand(op.operands[2])
    self.assign(
        result,
        f"({condition} ? {true_value} : {false_value})",
    )
```

So the leaky-ReLU select:

```text
%9 = select %7, %8, %6 : vector<256 x f32>
```

becomes:

```cuda
float v9 = (v7 ? v8 : v6);
```

That is a value expression, not a CUDA `if` statement. The store is still
guarded by the original memory mask. The select only decides which value this
thread wants to store.

## Minimum and maximum are not just pretty comparisons

At first glance, minimum and maximum seem like tiny wrappers around `<` and
`>`:

```cuda
lhs < rhs ? lhs : rhs
lhs > rhs ? lhs : rhs
```

For integers, that is basically the whole story.

For floating-point values, it is not. `NaN` and signed zero make the operation
more interesting.

Version 4 lowers floating-point extrema like this:

```python
elif op.opcode in ("maximum", "minimum"):
    lhs = self.expression_operand(op.operands[0])
    rhs = self.expression_operand(op.operands[1])
    symbol = ">" if op.opcode == "maximum" else "<"
    comparison = f"(({lhs}) {symbol} ({rhs}) ? ({lhs}) : ({rhs}))"
    if self.scalar_type(result.ty) == F32:
        comparison = (
            f"(isnan({lhs}) ? ({lhs}) : "
            f"(isnan({rhs}) ? ({rhs}) : {comparison}))"
        )
    self.assign(result, comparison)
```

There are two explicit choices here.

First, `NaN` propagates. If the left operand is `NaN`, the result is the left
operand. If the right operand is `NaN`, the result is the right operand.

Second, equal values choose the right-hand operand. The comparison uses `<` for
minimum and `>` for maximum:

```cuda
(lhs < rhs ? lhs : rhs)
(lhs > rhs ? lhs : rhs)
```

If the values compare equal, the condition is false and the result is `rhs`.
That detail matters for signed zero:

```text
minimum( 0.0, -0.0) -> -0.0
minimum(-0.0,  0.0) ->  0.0
maximum( 0.0, -0.0) -> -0.0
maximum(-0.0,  0.0) ->  0.0
```

This is one of those places where a tiny compiler becomes more educational than
a tiny amount of code would suggest. The operation name looks simple, but the
lowering has to decide real floating-point behavior.

I could have printed `fminf` and `fmaxf`. Instead, Version 4 spells out the
semantics it wants. The generated CUDA is more verbose, but it is also easier
to inspect.

For ReLU, the key generated line becomes:

```cuda
float v7 = (isnan(v6) ? (v6) : (isnan(0.0f) ? (0.0f) : ((v6) > (0.0f) ? (v6) : (0.0f))));
```

It is not beautiful CUDA, but it is honest CUDA. It says exactly how the
compiler currently defines `tl.maximum(a, 0.0)`.

## ReLU as a small activation kernel

The complete ReLU SSA is still mostly the familiar vector-add skeleton:

```text
%0 = program_id {axis=0} : i32
%1 = mul %0, 256 : i32
%2 = arange {start=0, end=256} : vector<256 x i32>
%3 = add %1, %2 : vector<256 x i32>
%4 = addptr x, %3 : vector<256 x ptr<f32>>
%5 = cmp_lt %3, n : vector<256 x bool>
%6 = load %4, %5, 0.0 : vector<256 x f32>
%7 = maximum %6, 0.0 : vector<256 x f32>
%8 = addptr out, %3 : vector<256 x ptr<f32>>
store %8, %7, %5
```

The only new value is `%7`.

That is exactly why ReLU is a useful first activation kernel. It does not force
the compiler to learn a new memory pattern. It isolates the new feature:
elementwise maximum.

The CUDA has the same shape as the vector-add kernel from
[Part 3: CUDA Lowering]({% post_url 2026-06-24-My_Triton_From_Scratch_Part_3_CUDA_Lowering %}):

```cuda
extern "C" __global__
void relu_kernel(float* x, float* out, int n) {
    int v0 = blockIdx.x;
    int v1 = (v0 * 256);
    int v2 = threadIdx.x;
    int v3 = (v1 + v2);
    bool v5 = (v3 < n);
    float v6 = (v5 ? x[v3] : 0.0f);
    float v7 = (isnan(v6) ? (v6) : (isnan(0.0f) ? (0.0f) : ((v6) > (0.0f) ? (v6) : (0.0f))));
    if (v5) {
        out[v3] = v7;
    }
}
```

The load mask and store mask are unchanged. The one-lane-per-thread mapping is
unchanged. The only new target idea is the expression that computes `v7`.

I like milestones like this because they are narrow in the right direction. The
new feature is easy to locate in every representation.

## Leaky ReLU as value selection

Leaky ReLU adds one more idea on top of ReLU. Instead of clamping negative
values to zero, it keeps a small negative slope:

```python
b = tl.where(a < 0.0, alpha * a, a)
```

The complete SSA shows three new operations after the load:

```text
%0 = program_id {axis=0} : i32
%1 = mul %0, 256 : i32
%2 = arange {start=0, end=256} : vector<256 x i32>
%3 = add %1, %2 : vector<256 x i32>
%4 = addptr x, %3 : vector<256 x ptr<f32>>
%5 = cmp_lt %3, n : vector<256 x bool>
%6 = load %4, %5, 0.0 : vector<256 x f32>
%7 = cmp_lt %6, 0.0 : vector<256 x bool>
%8 = mul alpha, %6 : vector<256 x f32>
%9 = select %7, %8, %6 : vector<256 x f32>
%10 = addptr out, %3 : vector<256 x ptr<f32>>
store %10, %9, %5
```

There are two masks in this kernel now, and they mean different things.

`%5` is the memory mask. It protects the tail of the vector so out-of-range
lanes do not load or store.

`%7` is the value-selection mask. It comes from `a < 0.0` and chooses the leaky
path for negative values.

That distinction is exactly why `select` is useful. Without it, every Boolean
vector in the language would be tied to memory guarding. With it, Boolean
vectors can drive ordinary computation.

The generated CUDA makes the difference visible:

```cuda
extern "C" __global__
void leaky_relu_kernel(float* x, float* out, int n, float alpha) {
    int v0 = blockIdx.x;
    int v1 = (v0 * 256);
    int v2 = threadIdx.x;
    int v3 = (v1 + v2);
    bool v5 = (v3 < n);
    float v6 = (v5 ? x[v3] : 0.0f);
    bool v7 = (v6 < 0.0f);
    float v8 = (alpha * v6);
    float v9 = (v7 ? v8 : v6);
    if (v5) {
        out[v3] = v9;
    }
}
```

The `select` became `v7 ? v8 : v6`. The store still uses `if (v5)`.

That is the compiler boundary I wanted: selection is not a side effect, and a
masked store is not just a value expression. They are separate operations with
separate lowering rules.

## Sigmoid as composition

Sigmoid is a better probe of composition.

The source is still small:

```python
b = 1.0 / (1.0 + tl.exp(-a))
```

but it becomes five SSA operations:

```text
%7 = neg %6 : vector<256 x f32>
%8 = exp %7 : vector<256 x f32>
%9 = add 1.0, %8 : vector<256 x f32>
%10 = div 1.0, %9 : vector<256 x f32>
store %11, %10, %5
```

The scalar constants `1.0` are not vector values in the expression tree, but
type inference broadcasts them when they meet a vector. CUDA lowering then
prints one scalar expression per thread:

```cuda
float v7 = -(v6);
float v8 = expf(v7);
float v9 = (1.0f + v8);
float v10 = (1.0f / v9);
```

That little sequence is the payoff for the previous layers.

The frontend does not need to understand sigmoid. Type inference does not need
to know it is computing an activation function. SSA lowering does not need a
sigmoid opcode. CUDA codegen does not need a sigmoid special case.

Each layer only knows its local rules:

- unary negation preserves numeric type and shape;
- `exp` accepts `f32` and preserves shape;
- arithmetic broadcasts scalar constants over vectors;
- CUDA emits one scalar operation per lane.

Together, those local rules produce a useful kernel.

## A small language starts to feel like a language

The first three versions were mostly about building the path:

```text
Python function -> expression tree -> typed SSA -> CUDA C++ -> GPU execution
```

Version 4 starts using that path for new vocabulary.

That is a different feeling. Adding `tl.maximum` is not just adding a Python
function. It means the tracer can capture it, the type system can reason about
it, SSA can name it, CUDA can lower it, and the resulting kernel can run.

Adding `tl.exp` is the same. Adding `tl.where` is the same. Adding unary
negation is the same.

The compiler is still tiny, but the layers are beginning to pay rent. Once the
pipeline exists, a new operation becomes a sequence of small, checkable
extensions rather than a new one-off translator.

That is the pattern I wanted for mytriton. Not a large compiler yet, but a
compiler-shaped place where language features have to become explicit at every
level.

## What's next

The next useful step is to make the IR a little more self-checking.

By this point, mytriton has enough opcodes that a small SSA verifier would be
worth having. It could check definition order, operand categories, result
types, and backend legality before CUDA source generation begins.

Once there is a verifier, a tiny optimization pipeline would also start to make
sense: constant folding, dead-code elimination, and simple rewrites around
`select` and masks. I do not want a large optimizer yet, but I do want a place
where these transformations can live deliberately instead of being hidden inside
code generation.

All code for this milestone is available at
[https://github.com/pbelevich/mytriton/tree/ver4](https://github.com/pbelevich/mytriton/tree/ver4).

Next: [Part 5: Verification]({% post_url 2026-06-26-My_Triton_From_Scratch_Part_5_Verification %}).
