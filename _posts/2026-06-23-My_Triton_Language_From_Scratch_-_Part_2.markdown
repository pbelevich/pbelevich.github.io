---
layout: post
title:  "My Triton Language From Scratch Part 2"
date:   2026-06-23 12:00:00 +0000
# categories:
---

In [Part 1]({% post_url 2026-06-22-My_Triton_Language_From_Scratch_-_Part_1 %}),
mytriton learned how to take a small Triton-style Python kernel and turn it
into compiler data.

That first version deliberately stopped at an expression-tree IR. Python
operators created nodes such as `BinOp`, pointer arithmetic created `AddPtr`,
and the final `tl.store` held one large tree containing everything needed to
compute its value.

It was enough to answer the first question: how does an ordinary-looking Python
function stop being ordinary Python and start describing a program?

But the resulting program was not yet in a particularly useful compiler form.
Repeated values were hidden inside a nested tree. Types existed on parameters,
but were not propagated through operations. There was no compact way to say
"this operation uses the result of that earlier operation."

Version 2 adds that missing layer. It infers types for the expression tree and
lowers it into a small typed
[SSA-style IR](https://en.wikipedia.org/wiki/Static_single-assignment_form).

The code for this milestone is here:
[https://github.com/pbelevich/mytriton/tree/ver2](https://github.com/pbelevich/mytriton/tree/ver2).

There is still no GPU code generation. The kernel still does not execute. But
the program is now explicit, linear, typed, and much closer to the form that
later compiler passes will want to consume.

## The expression tree we started with

The kernel is the same vector addition from
[Part 1]({% post_url 2026-06-22-My_Triton_Language_From_Scratch_-_Part_1 %}):

```python
@triton.jit
def add_kernel(x, y, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n

    x_values = tl.load(x + offsets, mask=mask, other=0.0)
    y_values = tl.load(y + offsets, mask=mask, other=0.0)

    tl.store(out + offsets, x_values + y_values, mask=mask)
```

The frontend from Version 1 captures something conceptually like this:

```text
Store(
  ptr=AddPtr(out, offsets),
  value=BinOp(
    "+",
    Load(AddPtr(x, offsets), mask, 0.0),
    Load(AddPtr(y, offsets), mask, 0.0),
  ),
  mask=mask,
)
```

This representation is convenient to build. Each overloaded Python operator
simply wraps its operands in another node.

It is less convenient to analyze.

If I want to inspect the type of `x_values`, I have to recursively understand
the pointer expression, its vector offsets, the load mask, and the fallback
value. If I want to print operations in execution order, I have to walk the
tree. If I later want to fold constants or eliminate unused computations,
there are no explicit operation results to rewrite.

The tree also visually repeats `offsets` and `mask` several times, even though
the tracer is reusing the same expression objects created by the Python
variables.

That is exactly the sort of problem SSA is good at making boring.

## What SSA gives us

SSA stands for Static Single Assignment. The central rule is that every SSA
value is defined exactly once.

Instead of nesting one expression inside another, each operation produces a
new value:

```text
%1 = mul %0, 256
%2 = arange 0, 256
%3 = add %1, %2
```

Later operations refer to those values by name:

```text
%4 = addptr x, %3
%5 = cmp_lt %3, n
```

`%3` is defined once and can be used as many times as necessary. There is no
assignment that later changes what `%3` means.

In a compiler with branches and loops, SSA also needs basic blocks and a way to
merge values arriving from different control-flow paths, usually phi nodes or
[block arguments](https://mlir.llvm.org/docs/LangRef/). mytriton does not have
control flow yet, so Version 2 only needs the straight-line part of SSA.

That still gives us the most important immediate benefits:

- every intermediate result has an explicit identity;
- operation order is visible;
- shared expression nodes are lowered once;
- every result carries a type;
- later passes can refer to and replace individual values.

## A small type system

Before producing typed SSA values, the compiler needs types to attach to them.

The type system is intentionally tiny:

```python
@dataclass(frozen=True)
class ScalarType:
    name: str


@dataclass(frozen=True)
class PointerType:
    element: ScalarType
    address_space: str = "global"


@dataclass(frozen=True)
class VectorType:
    size: int
    element: ScalarType | PointerType
```

There are currently three scalar types:

```python
I32 = ScalarType("i32")
F32 = ScalarType("f32")
BOOL = ScalarType("bool")
```

And one alias describing every type inference can return:

```python
Type = ScalarType | PointerType | VectorType
```

The interesting detail is that a vector element can itself be a pointer.

These two types mean very different things:

```text
vector<256 x f32>
vector<256 x ptr<f32>>
```

The first is 256 floating-point values. The second is 256 addresses, each
pointing to an `f32` value.

That distinction appears naturally in the add kernel. The argument `x` starts
as one base pointer:

```text
x : ptr<f32>
```

The offsets are a vector:

```text
offsets : vector<256 x i32>
```

Adding the offsets to the base pointer creates one address per lane:

```text
x + offsets : vector<256 x ptr<f32>>
```

Loading through those addresses produces the values stored there:

```text
load(x + offsets) : vector<256 x f32>
```

Representing both as an unqualified "vector" would throw away exactly the
information needed to reject loading from numbers or doing arithmetic directly
on addresses.

## Inferring constants and parameters

Type inference starts with the leaves of the tree.

Constants get their type from their Python value:

```python
if isinstance(expr, Const):
    if isinstance(expr.value, bool):
        ty = BOOL
    elif isinstance(expr.value, int):
        ty = I32
    elif isinstance(expr.value, float):
        ty = F32
    else:
        raise TypeError(f"Unsupported constant: {expr.value!r}")
```

The Boolean check comes first because `bool` is a subclass of `int` in Python.
Without that ordering, `True` would quietly become `i32`.

Runtime parameters already received a type when tracing began, so inference can
return it directly:

```python
elif isinstance(expr, Param):
    ty = expr.ty
```

`program_id` is currently an `i32`, and `arange` produces a vector of `i32`:

```python
elif isinstance(expr, ProgramId):
    ty = I32

elif isinstance(expr, Arange):
    size = expr.end - expr.start

    if size <= 0:
        raise TypeError(
            f"arange requires end > start, got [{expr.start}, {expr.end})"
        )

    ty = VectorType(size, I32)
```

This is already more than annotation. Inference is beginning to verify the
program as well. An invalid range does not get to become an invalid vector type.

## Promotion and broadcasting

Binary operations need to answer two separate questions:

1. What is the element type of the result?
2. Is the result a scalar or a vector?

I keep those decisions separate.

Numeric promotion chooses the element type:

```python
def promote(self, lhs: Type, rhs: Type) -> ScalarType:
    lhs_element = self.element_type(lhs)
    rhs_element = self.element_type(rhs)

    if lhs_element not in (I32, F32) or rhs_element not in (I32, F32):
        raise TypeError(f"Cannot combine {lhs} and {rhs}")

    return F32 if F32 in (lhs_element, rhs_element) else I32
```

For now, combining `i32` with `f32` produces `f32`. Combining two `i32` values
produces `i32`. Pointer and Boolean arithmetic are rejected.

Broadcasting chooses the shape:

```python
def common_vector_size(self, *types: Type) -> int | None:
    sizes = {ty.size for ty in types if isinstance(ty, VectorType)}

    if len(sizes) > 1:
        rendered = ", ".join(str(ty) for ty in types)
        raise TypeError(f"Cannot broadcast: {rendered}")

    return next(iter(sizes), None)
```

Scalars do not contribute a vector size. One vector and any number of scalars
produce that vector's shape. Two vectors must have the same size.

The helper that puts the two pieces back together is small:

```python
def with_shape(
    self,
    element: ScalarType | PointerType,
    *types: Type,
) -> Type:
    size = self.common_vector_size(*types)
    return VectorType(size, element) if size is not None else element
```

As a result:

```text
i32 + i32                         -> i32
i32 + f32                         -> f32
vector<256 x i32> + i32           -> vector<256 x i32>
vector<256 x i32> + f32           -> vector<256 x f32>
vector<256 x i32> + vector<8 x i32> -> error
```

Comparisons use numeric promotion to verify their operands, but their result
element is Boolean:

```python
if expr.op == "<":
    self.promote(lhs, rhs)
    ty = self.with_shape(BOOL, lhs, rhs)
```

That is how `offsets < n` becomes `vector<256 x bool>`.

## Pointer arithmetic has its own rules

[Part 1]({% post_url 2026-06-22-My_Triton_Language_From_Scratch_-_Part_1 %})
introduced `AddPtr` because pointer addition is not ordinary numeric addition.
Version 2 finally uses that distinction for type checking.

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

The base must contain pointers. The offset must contain `i32`. Broadcasting then
decides whether the result is one pointer or a vector of pointers.

For the add kernel:

```text
ptr<f32> + vector<256 x i32>
    -> vector<256 x ptr<f32>>
```

This is a small verifier, but it already prevents several nonsense programs
from moving further into the compiler.

## Loads combine pointer, mask, and fallback shapes

A load is slightly more interesting because three operands can affect its
shape:

- the pointer or vector of pointers;
- the optional mask;
- the optional fallback value called `other`.

The pointer determines the result element type:

```python
ptr = self.infer(expr.ptr)
ptr_element = self.element_type(ptr)

if not isinstance(ptr_element, PointerType):
    raise TypeError(f"Cannot load from {ptr}")
```

The mask must contain Boolean values:

```python
if expr.mask is not None:
    mask = self.infer(expr.mask)
    self.require_mask(mask)
    operands.append(mask)
```

And the fallback must be convertible to the loaded element type:

```python
if expr.other is not None:
    other = self.infer(expr.other)
    self.require_convertible(
        other,
        ptr_element.element,
        context="Load fallback",
    )
    operands.append(other)
```

Finally, all participating operands must broadcast to one shape:

```python
ty = self.with_shape(ptr_element.element, *operands)
```

This means even a scalar pointer combined with a vector mask produces a vector
load in the current model. Conceptually, the scalar address is broadcast across
the lanes.

The conversion rules are deliberately narrow. Equal element types are valid,
and `i32` and `f32` can convert between each other. Boolean values only convert
to Boolean values.

```python
if source_element == destination:
    return

numeric = (I32, F32)
if source_element in numeric and destination in numeric:
    return

raise TypeError(...)
```

The goal is not to reproduce every
[Triton conversion rule](https://triton-lang.org/main/python-api/triton-semantics.html)
yet. It is to make the rules explicit instead of allowing Python values to
drift through the IR unchecked.

## Stores need verification even though they produce no value

Type inference naturally returns a type for expressions. A store is different:
it is a side effect and has no result.

It still needs verification.

```python
def check_store(self, store: Store) -> None:
    ptr = self.infer(store.ptr)
    value = self.infer(store.value)
    ptr_element = self.element_type(ptr)

    if not isinstance(ptr_element, PointerType):
        raise TypeError(f"Cannot store to {ptr}")

    self.require_convertible(
        value,
        ptr_element.element,
        context="Stored value",
    )

    operands = [ptr, value]

    if store.mask is not None:
        mask = self.infer(store.mask)
        self.require_mask(mask)
        operands.append(mask)

    self.common_vector_size(*operands)
```

This checks the destination, the stored value, the mask, and their shapes. A
store does not need an SSA result to participate in a typed IR.

That separation also hints at a useful compiler distinction: some operations
compute values, some operations produce side effects, and some eventually do
both.

## Representing SSA values and operations

The SSA data model is almost aggressively small:

```python
@dataclass(frozen=True)
class SSAValue:
    id: int
    ty: Type

    def __str__(self):
        return f"%{self.id}"
```

An SSA value is an identity and a type. The operation that defines it lives in
the operation list.

```python
SSAOperand = SSAValue | Param | Const | None


@dataclass
class SSAOp:
    opcode: str
    operands: tuple[SSAOperand, ...] = ()
    result: SSAValue | None = None
    attrs: dict[str, object] = field(default_factory=dict)
```

Operands can be earlier SSA results, kernel parameters, constants, or `None`
for omitted optional operands such as an absent load mask.

Attributes describe compile-time properties of an operation. For example,
`program_id` stores its axis as an attribute, while `arange` stores its start
and end.

Parameters and constants do not receive numbered SSA definitions in this
version. They enter the IR as external operands. A future function-level IR can
make kernel parameters explicit in a typed function signature.

## Lowering the tree in dependency order

Lowering recursively visits each expression's operands before emitting the
expression itself. That is a post-order traversal of the tree.

For a binary operation:

```python
if isinstance(expr, BinOp):
    if expr.op not in self.BINOPS:
        raise TypeError(f"Unsupported binary operator: {expr.op}")

    lhs = self.lower_expr(expr.lhs)
    rhs = self.lower_expr(expr.rhs)

    return self.emit(
        self.BINOPS[expr.op],
        expr,
        operands=(lhs, rhs),
    )
```

The mapping turns frontend syntax into SSA opcodes:

```python
BINOPS = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
    "<": "cmp_lt",
}
```

This is a useful boundary. The expression tree remembers that Python used `<`.
The SSA IR says that the compiler operation is `cmp_lt`.

Emission asks type inference for the result type and allocates the next SSA
identity:

```python
def new_result(self, expr):
    result = SSAValue(
        id=self.next_id,
        ty=self.type_inference.infer(expr),
    )
    self.next_id += 1
    return result
```

Then it records the operation:

```python
def emit(self, opcode, expr, operands=(), attrs=None):
    result = self.new_result(expr)

    self.ops.append(
        SSAOp(
            opcode=opcode,
            operands=tuple(operands),
            result=result,
            attrs=attrs or {},
        )
    )

    self.memo[id(expr)] = result
    return result
```

The important line is the memo table.

## Lowering shared expressions once

Python variables in the traced kernel reuse expression objects. `offsets` is
created once, then referenced by both pointer additions, the output pointer,
and the comparison that creates `mask`.

Before lowering an expression, the compiler checks whether that exact node has
already produced an SSA value:

```python
if id(expr) in self.memo:
    return self.memo[id(expr)]
```

The first time `offsets` is visited, lowering emits `%3`. Every later use gets
the same `%3` back. The same happens for the mask in `%5`.

This is identity-based sharing, not a general common-subexpression elimination
pass. Two separately constructed but structurally identical `BinOp` objects
would still become two SSA operations. That is intentional. The lowering pass
preserves sharing already present in the input graph; an optimizer can discover
additional equivalences later.

## Side effects stay in the operation list

Top-level operations are currently stores. Before lowering one, the compiler
verifies its complete expression tree:

```python
if isinstance(op, Store):
    self.type_inference.check_store(op)

    value = self.lower_expr(op.value)
    ptr = self.lower_expr(op.ptr)
    mask = self.lower_expr(op.mask) if op.mask is not None else None

    self.ops.append(
        SSAOp(
            opcode="store",
            operands=(ptr, value, mask),
        )
    )
```

The store has no `result`, but it remains in sequence with the value-producing
operations. This becomes important as soon as the language has multiple memory
effects: operation order is part of program meaning.

## The complete SSA for vector addition

After tracing and lowering, the launch now returns both representations:

```python
expression_ops, ssa_ops = add_kernel[grid](
    x,
    y,
    out,
    n,
    BLOCK=256,
)
```

The SSA printer produces:

```text
%0 = program_id {axis=0} : i32
%1 = mul %0, 256 : i32
%2 = arange {start=0, end=256} : vector<256 x i32>
%3 = add %1, %2 : vector<256 x i32>
%4 = addptr x, %3 : vector<256 x ptr<f32>>
%5 = cmp_lt %3, n : vector<256 x bool>
%6 = load %4, %5, 0.0 : vector<256 x f32>
%7 = addptr y, %3 : vector<256 x ptr<f32>>
%8 = load %7, %5, 0.0 : vector<256 x f32>
%9 = add %6, %8 : vector<256 x f32>
%10 = addptr out, %3 : vector<256 x ptr<f32>>
store %10, %9, %5
```

This is the same program as the nested tree, but several things are now
immediately visible.

`%3`, the offsets vector, is computed once and reused for `x`, `y`, `out`, and
the mask. `%5`, the Boolean mask, is computed once and reused by both loads and
the store.

Pointer arithmetic produces vectors of addresses:

```text
%4 : vector<256 x ptr<f32>>
```

Loads turn those addresses into vectors of values:

```text
%6 : vector<256 x f32>
```

And the final numeric addition can only combine the two value vectors:

```text
%9 = add %6, %8 : vector<256 x f32>
```

The types are no longer comments I have to keep in my head. They are part of
the IR.

## Why add this layer before execution?

It would be possible to write an interpreter that recursively evaluates the
Version 1 expression tree. That might even be fewer lines of code in the short
term.

But doing so would skip the representation where most compiler work becomes
manageable.

Execution needs to know the order of operations. Optimization needs explicit
inputs and outputs. Diagnostics need a place to attach type errors. Code
generation needs a predictable set of typed opcodes. SSA gives all of those
future stages one common language.

It also forces vague assumptions to become decisions.

Is `x + offsets` numeric addition or pointer arithmetic? What shape does a
masked load produce? Can an integer fallback feed a floating-point load? Does a
store produce a value? What happens when two vector sizes disagree?

The expression tree allowed some of those questions to remain hidden. Typed SSA
does not.

That is one reason I like building compiler layers in this order. Each layer is
small, but each one removes a category of ambiguity from the next.

## What I want to build next

The next useful milestone is lowering this SSA IR to CUDA and running the
result on a GPU.

I want to add a small CUDA backend that translates each typed SSA operation into
CUDA C. `program_id` can become a CUDA block index, vector lanes can be mapped
to CUDA threads, `addptr` can become pointer arithmetic, and masked loads and
stores can become ordinary guarded memory accesses.

CuPy will provide the bridge from generated source code to execution. It can
compile the emitted CUDA C with NVRTC, hold the input and output arrays on the
GPU, and launch the resulting kernel. That keeps the next version focused on
lowering semantics instead of requiring mytriton to implement a CUDA driver
layer as well.

This will make the complete path concrete:

```text
Python kernel
    -> expression-tree IR
    -> typed SSA IR
    -> CUDA C
    -> CuPy/NVRTC
    -> GPU execution
```

The new backend will also force another set of decisions into the open. How
should a vector SSA value map to individual CUDA threads? How should the launch
grid determine `blockIdx` and `threadIdx`? Which type conversions need explicit
CUDA casts? How should masked memory operations avoid out-of-bounds accesses?

Those are exactly the questions I want the next version to answer. Typed SSA
now gives the CUDA lowering pass a compact, ordered, and verified input instead
of asking code generation to rediscover the structure hidden inside an
expression tree.

For now, Version 2 can take a nested symbolic program, infer what every
operation means, reject several invalid programs, and turn the result into a
compact typed operation sequence.

The program still does not run, but it has become much easier to reason about.
That feels like a very compiler-shaped kind of progress.

All code for this milestone is available at
[https://github.com/pbelevich/mytriton/tree/ver2](https://github.com/pbelevich/mytriton/tree/ver2).

Next: [Part 3]({% post_url 2026-06-24-My_Triton_Language_From_Scratch_-_Part_3 %}).
