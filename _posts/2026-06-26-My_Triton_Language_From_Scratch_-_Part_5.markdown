---
layout: post
title:  "My Triton Language From Scratch Part 5"
date:   2026-06-26 14:00:00 +0000
# categories:
---

In Part 1, mytriton learned how to trace a Python function into an
expression-tree IR.

In Part 2, it learned how to infer types and lower that tree into typed
[SSA-style](https://en.wikipedia.org/wiki/Static_single-assignment_form)
operations.

In Part 3, it learned how to turn those SSA operations into CUDA C++ and launch
the generated kernel.

In Part 4, the language grew enough elementwise operations to write ReLU, leaky
ReLU, and sigmoid: unary negation, `tl.exp`, `tl.minimum`, `tl.maximum`, and
`tl.where`.

Version 5 does not add a flashy new kernel. Instead, it adds something that
starts to matter once the compiler has enough moving parts:

- an SSA verifier;
- a small optimization pipeline;
- verifier checks after every optimization pass.

The code for this milestone is here:
[https://github.com/pbelevich/mytriton/tree/ver5](https://github.com/pbelevich/mytriton/tree/ver5).

This is a quieter milestone than generating CUDA for the first time, but it is
one of my favorite kinds of compiler work. The project stops trusting that every
previous layer did the right thing, and starts checking the contract between
layers explicitly.

The pipeline now looks like this:

```text
Python function
    -> expression tree
    -> typed SSA
    -> SSA verifier
    -> optimization pipeline
    -> SSA verifier after each pass
    -> CUDA C++
```

The generated CUDA is still the final visible artifact. But Version 5 is about
making the middle of the compiler more deliberate.

## Why add a verifier now?

In the first few versions, a verifier would have been a little ceremonial.

Version 1 only had an expression tree. Version 2 had a small number of SSA
operations. Version 3 immediately consumed that SSA in the CUDA backend. Version
4 changed the feeling: the IR now has arithmetic, comparisons, pointer
arithmetic, loads, stores, extrema, unary operations, and `select`.

At that point, there are more ways for an operation list to be malformed:

```text
%0 = exp 1 : i32
```

`exp` is only supposed to operate on `f32`.

```text
%0 = add 1, 2 : i32
store out, %1, none
```

`%1` was never defined.

```text
%0 = select 1, 2.0, 3.0 : f32
```

The condition of `select` is not Boolean.

The normal frontend should not produce these examples. But a compiler IR is not
only consumed by the frontend. Optimization passes will rewrite it. Future
lowering code might construct it directly. Tests might build it by hand.

The verifier gives the compiler a clear boundary: before optimization and code
generation, SSA has to prove that it is well-formed.

## The shape of the verifier

The verifier starts with the most mechanical rule: every opcode has a fixed
arity.

```python
class SSAVerifier:
    ARITY: ClassVar[dict[str, int]] = {
        "program_id": 0,
        "arange": 0,
        "add": 2,
        "sub": 2,
        "mul": 2,
        "div": 2,
        "cmp_lt": 2,
        "neg": 1,
        "exp": 1,
        "maximum": 2,
        "minimum": 2,
        "addptr": 2,
        "load": 3,
        "store": 3,
        "select": 3,
    }
```

This table is intentionally conservative. It only contains operations that the
current compiler can actually lower to CUDA. I do not want the verifier to
accept future-looking operations that the backend would reject later.

Each invalid program fails with a `CompileError` that points at the offending
operation:

```python
def fail(self, index: int, op: SSAOp, message: str) -> NoReturn:
    raise CompileError(f"ssa-verifier: op #{index} '{op.opcode}': {message}")
```

The error messages are not fancy, but they are close to the layer where the
mistake happened. That is already a big improvement over letting a malformed
operation trickle down into CUDA string generation.

## Def-before-use and result declarations

SSA has a simple promise: a value is defined once before it is used.

mytriton checks that while walking the operation list:

```python
defined: set[int] = set()

for index, op in enumerate(ops):
    for operand in op.operands:
        if isinstance(operand, SSAValue) and operand.id not in defined:
            self.fail(index, op, f"{operand} used before definition")

    if op.result is not None:
        if op.result.id in defined:
            self.fail(index, op, f"duplicate definition of {op.result}")

        defined.add(op.result.id)
```

The verifier also checks whether an operation should have a result at all:

```python
should_have_result = op.opcode != "store"

if should_have_result != (op.result is not None):
    self.fail(index, op, "invalid result declaration")
```

For now, `store` is the only side-effecting operation and it does not produce a
value. Everything else is value-producing.

This rule is tiny, but it is exactly the sort of rule I want outside the CUDA
backend. CUDA generation should not have to rediscover whether `store` returns
something. That belongs to the IR contract.

## Type rules at the SSA boundary

Part 2 already had type inference. So why repeat type checks in a verifier?

Because they protect different boundaries.

Type inference works on the expression tree and helps construct typed SSA.
The verifier works on the SSA itself. If SSA comes from the normal frontend, the
checks should agree. If SSA is malformed, hand-written in a test, or broken by
an optimization pass, the verifier catches it before code generation.

The verifier has the same small type vocabulary:

```python
def element_type(self, ty: Type) -> ScalarType | PointerType:
    return ty.element if isinstance(ty, VectorType) else ty


def vector_size(self, index: int, op: SSAOp, *types: Type) -> int | None:
    sizes = {ty.size for ty in types if isinstance(ty, VectorType)}

    if len(sizes) > 1:
        rendered = ", ".join(str(ty) for ty in types)
        self.fail(index, op, f"incompatible shapes: {rendered}")

    return next(iter(sizes), None)
```

The `vector_size` helper is the verifier version of scalar-vector broadcasting.
All vector operands in one operation must either have the same width or be
scalars.

Arithmetic and comparisons use numeric promotion:

```python
def check_binary_numeric(self, index: int, op: SSAOp) -> None:
    lhs_ty = self.require_operand_type(index, op, op.operands[0], "lhs")
    rhs_ty = self.require_operand_type(index, op, op.operands[1], "rhs")
    result_ty = self.result_type(index, op)

    element = self.promote_numeric(index, op, lhs_ty, rhs_ty)
    if op.opcode == "cmp_lt":
        element = BOOL

    expected_ty = self.with_shape(index, op, element, lhs_ty, rhs_ty)
    self.require_type(index, op, result_ty, expected_ty)
```

So:

```text
add(i32, f32)                         -> f32
add(vector<256 x i32>, f32)           -> vector<256 x f32>
cmp_lt(vector<256 x i32>, i32)        -> vector<256 x bool>
```

`select` combines two rules: the condition must be Boolean, and the two value
arms use numeric promotion and broadcasting:

```python
def check_select(self, index: int, op: SSAOp) -> None:
    condition, true_value, false_value = op.operands
    condition_ty = self.require_operand_type(index, op, condition, "condition")
    true_ty = self.require_operand_type(index, op, true_value, "true value")
    false_ty = self.require_operand_type(index, op, false_value, "false value")
    result_ty = self.result_type(index, op)

    if self.element_type(condition_ty) != BOOL:
        self.fail(index, op, f"condition must be bool, got {condition_ty}")

    element = self.promote_numeric(index, op, true_ty, false_ty)
    expected_ty = self.with_shape(
        index,
        op,
        element,
        condition_ty,
        true_ty,
        false_ty,
    )
    self.require_type(index, op, result_ty, expected_ty)
```

That means a vector mask can select between scalar arms:

```text
select(vector<256 x bool>, f32, f32) -> vector<256 x f32>
```

This is the kind of rule that would be easy to assume in the backend and hard
to debug later. It belongs in the verifier.

## Memory operations are checked too

Loads and stores are where pointer types, value types, masks, and shapes meet.

A load requires a pointer:

```python
ptr_ty = self.require_operand_type(index, op, ptr, "pointer")
ptr_element = self.element_type(ptr_ty)

if not isinstance(ptr_element, PointerType):
    self.fail(index, op, f"expected pointer, got {ptr_ty}")
```

The result type must match the element type of that pointer, with vector shape
coming from the pointer, mask, and fallback value:

```python
expected_ty = self.with_shape(
    index,
    op,
    ptr_element.element,
    *shape_operands,
)
self.require_type(index, op, result_ty, expected_ty)
```

Stores are similar, except that mytriton allows numeric conversion from `i32`
to `f32`:

```python
if not self.is_convertible(value_ty, ptr_element.element):
    self.fail(
        index,
        op,
        f"stored value must be convertible to {ptr_element.element}, "
        f"got {value_ty}",
    )
```

The mask, if present, must be Boolean and broadcast-compatible with the pointer
and value.

This is still a small verifier, but it already knows enough to protect the CUDA
backend from the most important category errors.

## The verifier knows about CUDA block size

One rule in the verifier is backend-specific:

```python
if width != self.block_size:
    self.fail(
        index,
        op,
        f"range width {width} does not match "
        f"CUDA block size {self.block_size}",
    )
```

This is because Version 3 made a deliberate lowering choice:

```text
one SSA vector lane == one CUDA thread
```

If the SSA contains:

```text
%2 = arange {start=0, end=256} : vector<256 x i32>
```

then the CUDA launch needs 256 threads per block. A different vector width in
the same kernel would be a problem for the current backend.

The compiler computes the block size by scanning vector result types:

```python
def _cuda_threads_per_block(ssa_ops: list[SSAOp]) -> int:
    vector_widths = {
        op.result.ty.size
        for op in ssa_ops
        if op.result is not None and isinstance(op.result.ty, VectorType)
    }

    if not vector_widths:
        return 1

    if len(vector_widths) != 1:
        rendered = ", ".join(str(width) for width in sorted(vector_widths))
        raise ValueError(f"CUDA lowering requires one vector width, got: {rendered}")
```

This is not a general layout system. It is a small legality check for the
current backend. But that is exactly what a verifier should do: make the target
assumptions visible.

## Where the verifier sits in the compiler

The compiler now verifies SSA immediately after lowering:

```python
ssa_ops = SSALowering().lower(ops)
threads_per_block = _cuda_threads_per_block(ssa_ops)

# The optimizer assumes lowering produced valid SSA.
verifier = SSAVerifier(threads_per_block)
verifier.verify(ssa_ops)
```

That first check establishes the contract for optimization passes. A pass can
assume it receives valid SSA.

Then the pass manager runs each optimization and verifies the result:

```python
ssa_ops = PassManager(
    passes=[
        ConstantFoldPass(),
        CSEPass(),
        DCEPass(),
    ],
    verifier=verifier,
).run(ssa_ops)
```

The `PassManager` is deliberately tiny:

```python
class PassManager:
    def __init__(self, passes, verifier):
        self.passes = passes
        self.verifier = verifier

    def run(self, ops):
        for compiler_pass in self.passes:
            ops = compiler_pass.run(ops)
            self.verifier.verify(ops)

        return ops
```

This is not an advanced pass framework. It is a place where transformations
live deliberately. That is already better than hiding rewrites inside CUDA
codegen.

After optimization, the compiler recomputes the block size and verifies once
more:

```python
threads_per_block = _cuda_threads_per_block(ssa_ops)
SSAVerifier(threads_per_block).verify(ssa_ops)
```

This final check matters because optimization can remove vector operations. If
all vector values disappear, a scalar-only kernel should launch with one thread
per block.

## Constant folding and local simplification

The first optimization pass is called `ConstantFoldPass`, although it does a
little more than pure constant folding.

It can simplify a constant `select`:

```text
%0 = select true, x, 0.0 : f32
store out, %0, none
```

into:

```text
store out, x, none
```

The code is straightforward:

```python
if op.opcode == "select":
    condition, true_value, false_value = args

    if isinstance(condition, Const):
        return true_value if condition.value else false_value

    if operand_key(true_value) == operand_key(false_value):
        return true_value
```

The second rule handles:

```text
select condition, x, x -> x
```

The pass also has small algebraic identities:

```python
if (
    op.opcode in ("add", "sub")
    and self.is_integer_result(op)
    and isinstance(rhs, Const)
    and rhs.value == 0
):
    return lhs

if (
    op.opcode in ("mul", "div")
    and isinstance(rhs, Const)
    and rhs.value == 1
):
    return lhs
```

Notice the integer guard on `x + 0 -> x`.

For floating-point values, even obvious-looking rewrites can change behavior
around signed zero and NaNs. I want the optimizer to be boringly correct before
it is clever.

## Folding extrema without changing semantics

Version 4 gave `minimum` and `maximum` explicit floating-point behavior:

- NaNs propagate;
- equal values choose the right-hand operand.

That means the optimizer cannot simply call Python's `max` or `min` and hope
for the best.

The folding helper mirrors the CUDA lowering:

```python
def extremum(self, opcode, lhs, rhs, ty):
    if self.scalar_type(ty) == F32:
        lhs = float(lhs)
        rhs = float(rhs)

        if math.isnan(lhs):
            return lhs
        if math.isnan(rhs):
            return rhs

    if opcode == "maximum":
        return lhs if lhs > rhs else rhs

    return lhs if lhs < rhs else rhs
```

The equal case falls through to `rhs`, just like the generated CUDA comparison:

```text
maximum( 0.0, -0.0) -> -0.0
maximum(-0.0,  0.0) ->  0.0
minimum( 0.0, -0.0) -> -0.0
minimum(-0.0,  0.0) ->  0.0
```

This is one of those compiler details that looks tiny but carries a lot of
meaning. An optimization is only valid if it preserves the language semantics,
including the weird corners.

## Common subexpression elimination

The second pass is common subexpression elimination, or
[CSE](https://en.wikipedia.org/wiki/Common_subexpression_elimination).

The idea is simple: if the same pure operation appears twice with the same
inputs, the second operation can reuse the first result.

For example:

```text
%0 = add 1, 2 : i32
%1 = add 1, 2 : i32
store out, %1, none
```

can become:

```text
%0 = add 1, 2 : i32
store out, %0, none
```

The pass keeps a table of available pure operations:

```python
class CSEPass:
    PURE_OPS: ClassVar[set[str]] = {
        "program_id",
        "arange",
        "add",
        "sub",
        "mul",
        "div",
        "cmp_lt",
        "neg",
        "exp",
        "maximum",
        "minimum",
        "addptr",
        "select",
    }
```

`load` and `store` are not in that list. A real compiler can reason about memory
and prove when loads are redundant. mytriton is not there yet, so memory
operations stay conservative.

The key includes the opcode, operands, attributes, and result type:

```python
key = (
    op.opcode,
    tuple(self.operand_key(x) for x in operands),
    tuple(sorted(op.attrs.items())),
    result.ty,
)
```

There is one small but important detail in the operand key:

```python
def const_key(value: object) -> tuple[object, ...]:
    if isinstance(value, float):
        return (float, value.hex())

    return (type(value), value)
```

`0.0 == -0.0` in Python, but the compiler cares about signed zero. Using
`float.hex()` keeps those constants distinct.

## Dead-code elimination

The third pass is dead-code elimination.

The SSA list can contain operations whose results are never used. They are safe
to remove as long as they have no side effects.

The pass walks the operation list backwards:

```python
class DCEPass:
    SIDE_EFFECTS: ClassVar[set[str]] = {"store"}

    def run(self, ops):
        live = set()
        output = []

        for op in reversed(ops):
            needed = op.opcode in self.SIDE_EFFECTS or (
                op.result is not None and op.result.id in live
            )

            if not needed:
                continue

            if op.result is not None:
                live.discard(op.result.id)

            for operand in op.operands:
                if isinstance(operand, SSAValue):
                    live.add(operand.id)

            output.append(op)

        return list(reversed(output))
```

Stores are roots of liveness because they affect memory. If a store uses `%9`,
then `%9` is live. If `%9` uses `%8`, then `%8` is live, and so on.

Everything else that does not feed a side effect can disappear.

## The compiler output is now optimized SSA

The launch API still returns the same three things:

```python
expression_ops, ssa_ops, cuda_src = kernel[grid](...)
```

But there is a small semantic change: `ssa_ops` is now the optimized SSA, not
the raw lowering output.

That is the representation CUDA codegen receives. It is also the representation
tests inspect when checking compiled kernels.

For most kernels from earlier parts, the printed SSA does not change much. The
interesting examples are small synthetic ones:

```text
%0 = select true, x, 0.0 : f32
store out, %0, none
```

becomes:

```text
store out, x, none
```

and:

```text
%0 = add 1, 2 : i32
%1 = add 1, 2 : i32
store out, %1, none
```

becomes:

```text
%0 = add 1, 2 : i32
store out, %0, none
```

This is enough optimization to prove that the pipeline exists and that passes
can rewrite SSA safely. It is not trying to be a performance optimizer yet.

## Tests become more compiler-shaped

Version 5 also adds tests that construct SSA directly.

That is a useful shift. Earlier tests mostly exercised the full frontend:
write a kernel, trace it, lower it, generate CUDA. Those tests are still
important, but they are not always the best way to test malformed IR.

For the verifier, direct SSA tests are clearer:

```python
with pytest.raises(CompileError, match="%0 used before definition"):
    verify([SSAOp("store", (out, SSAValue(0, F32), None))])
```

For the optimizer, small hand-written sequences are better than inventing
awkward kernels:

```python
ops = [
    SSAOp("add", (Const(1), Const(2)), first),
    SSAOp("add", (Const(1), Const(2)), second),
    SSAOp("store", (out, second, None)),
]
```

That makes it easy to assert the exact before-and-after behavior of one pass.

The project now has tests for:

- verifier failures such as use-before-def, invalid result declarations,
  unsupported opcodes, bad `select` conditions, and invalid load fallbacks;
- constant folding and local simplification;
- CSE;
- DCE;
- `PassManager` verifying after every pass.

That last one is important. I want broken optimization rewrites to fail in the
optimization pipeline, not later in CUDA code generation.

## A compiler-shaped middle

The most satisfying part of Version 5 is not any single optimization. The
optimizations are intentionally small.

The satisfying part is that mytriton now has a compiler-shaped middle:

```text
SSA is not just printed.
SSA is verified.
SSA is rewritten.
SSA is verified again.
SSA is lowered to CUDA.
```

That changes how future features can be added.

If I add a new opcode, the verifier has to know its arity and type rules. If an
optimization rewrites it, tests can check the before-and-after SSA directly. If
CUDA codegen receives it, the backend can assume the operation has already
passed the IR contract.

This is still a tiny compiler, but the layers are becoming stricter with each
other. That is a good direction.

## What's next

The next feature I want is a multidimensional launch grid.

So far, `tl.program_id(0)` is the only program coordinate that really matters in
the examples. The CUDA backend already knows how to print `blockIdx.x`,
`blockIdx.y`, and `blockIdx.z`, but the language has not used that to express a
real two-dimensional kernel yet.

A good next milestone would be something like a matrix-style copy or transpose:

```python
pid_m = tl.program_id(0)
pid_n = tl.program_id(1)
```

That would force the compiler to make the grid model more explicit without
jumping all the way to reductions or matrix multiplication.

After that, the natural next step is the first reduction: row-wise sum, row-wise
maximum, and eventually softmax.

All code for this milestone is available at
[https://github.com/pbelevich/mytriton/tree/ver5](https://github.com/pbelevich/mytriton/tree/ver5).
