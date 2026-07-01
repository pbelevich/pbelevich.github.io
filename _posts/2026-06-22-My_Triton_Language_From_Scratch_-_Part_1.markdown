---
layout: post
title:  "My Triton Language From Scratch Part 1"
date:   2026-06-22 12:00:00 +0000
# categories:
---

I have wanted to understand Triton at a deeper level for a while. Not just how
to write a kernel, but what has to happen after Python sees `@triton.jit` and
before a GPU sees executable code.

The obvious way to answer that question was to build a tiny version myself.

So this is **mytriton**: a small, intentionally incomplete Triton-like language
that I am building one compiler layer at a time. The point is not to compete
with the real Triton compiler. The point is to make the hidden parts visible,
small enough to fit in one head, and easy to experiment with.

The code for this first milestone is here:
[https://github.com/pbelevich/mytriton/tree/ver1](https://github.com/pbelevich/mytriton/tree/ver1).

Version 1 does one thing: it takes a small Triton-style Python kernel and turns
it into an expression-tree IR. There is no GPU code generation yet. There is no
[LLVM](https://llvm.org/docs/LangRef.html),
[MLIR](https://mlir.llvm.org/docs/LangRef/), PTX, or even a CPU interpreter.
We are starting earlier than that, at the moment where ordinary Python stops
being ordinary Python and starts describing a program.

## The tiny kernel we want to understand

Here is the example I use throughout the project:

```python
import mytriton as triton
import mytriton.language as tl


@triton.jit
def add_kernel(x, y, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n

    x_values = tl.load(x + offsets, mask=mask, other=0.0)
    y_values = tl.load(y + offsets, mask=mask, other=0.0)

    tl.store(out + offsets, x_values + y_values, mask=mask)
```

If you have seen
[Triton](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
before, this should feel familiar. Each program instance gets an ID, builds a
vector of offsets, masks the final partial block, loads two vectors, adds them,
and stores the result.

The launch also looks Triton-like:

```python
n = 1_000
block = 256

x = np.random.randn(n).astype(np.float32)
y = np.random.randn(n).astype(np.float32)
out = np.empty_like(x)


def grid(meta):
    return (triton.cdiv(n, meta["BLOCK"]),)


ops = add_kernel[grid](x, y, out, n, BLOCK=block)
```

There is one important twist: `out` is unchanged. Version 1 does not run the
kernel. The result in `ops` is the program that mytriton captured.

## The trick: run Python with values that do not compute

The heart of this version is symbolic tracing.

Instead of calling `add_kernel` with the real integer `n=1000`, the tracer calls
it with something like this:

```python
Value(Param(name="n", ty=I32))
```

And instead of passing the real NumPy array `x`, it passes a symbolic pointer:

```python
Ptr(Param(name="x", ty=PointerType(F32)))
```

These objects look like values to Python, but their
[operators](https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types)
build IR nodes instead of doing arithmetic. When Python evaluates:

```python
offsets < n
```

it eventually calls this method:

```python
def __lt__(self, other):
    return Value(BinOp("<", self.expr, unwrap(other)))
```

No comparison happens. We create a `BinOp` that says, "when this kernel really
runs one day, compare these two operands."

The same idea handles `+`, `-`, `*`, and `/`:

```python
class Value:
    def __init__(self, expr):
        self.expr = expr

    def __add__(self, other):
        return Value(BinOp("+", self.expr, unwrap(other)))

    def __mul__(self, other):
        return Value(BinOp("*", self.expr, unwrap(other)))
```

I like this approach for a first version because it is almost suspiciously
small. We do not need a parser yet. We let Python parse the function, dispatch
the operators, and handle arguments for us. Our symbolic objects quietly record
what happens.

## IR nodes versus Python-facing wrappers

There are two kinds of objects in the implementation, and keeping them separate
turned out to be useful.

First, there are plain IR nodes:

```python
@dataclass
class BinOp:
    op: str
    lhs: Any
    rhs: Any


@dataclass
class Load:
    ptr: Any
    mask: Any | None
    other: Any | None


@dataclass
class Store:
    ptr: Any
    value: Any
    mask: Any | None
```

They are deliberately boring dataclasses. They describe the program and are
easy to print, compare, and test.

Then there are `Value` and `Ptr`, the wrappers Python interacts with. They know
how to overload operators and turn the result back into an IR node.

The small `unwrap` helper is the bridge between the two worlds:

```python
def unwrap(x):
    if isinstance(x, Value):
        return x.expr
    if isinstance(x, Ptr):
        return x.expr
    if isinstance(x, (int, float)):
        return Const(x)
    return x
```

This is not a sophisticated IR framework. That is exactly why it is useful
right now: I can see every object the frontend creates without digging through
layers of infrastructure.

## Pointer addition is not normal addition

My first instinct was to represent every `+` with the same `BinOp`. That works
for a demo, but it immediately throws away an important fact: `x + offsets` is
still a pointer.

Version 1 has an explicit pointer operation instead:

```python
@dataclass
class AddPtr:
    base: Any
    offset: Any


class Ptr:
    def __init__(self, expr):
        self.expr = expr

    def __add__(self, offset):
        return Ptr(AddPtr(self.expr, unwrap(offset)))
```

That small distinction will matter much more later, when the compiler has to
verify types and lower pointer arithmetic correctly. It is also a nice early
example of a recurring compiler lesson: two operations that share Python syntax
do not necessarily share compiler semantics.

## Building language operations

The public language API is tiny. `program_id` and `arange` simply wrap new IR
nodes:

```python
def program_id(axis: int):
    return Value(ProgramId(axis))


def arange(start: int, end: int):
    return Value(Arange(start, end))
```

A load creates a value that can feed another expression:

```python
def load(ptr, mask=None, other=None):
    node = Load(
        ptr=unwrap(ptr),
        mask=unwrap(mask) if mask is not None else None,
        other=unwrap(other) if other is not None else None,
    )
    return Value(node)
```

A store is different because it is a side effect. It does not return a symbolic
result. It is appended to the current builder:

```python
def store(ptr, value, mask=None):
    node = Store(
        ptr=unwrap(ptr),
        value=unwrap(value),
        mask=unwrap(mask) if mask is not None else None,
    )
    Builder.current().ops.append(node)
```

This means the top-level IR is currently a list of side-effecting operations.
The arithmetic, comparisons, and loads are nested inside them as expression
trees.

It is simple, although not especially compact. The same `offsets` expression
appears several times in the final tree. That duplication is one of the reasons
I want to move to SSA in a later version.

## Where the operations go

The builder is a context manager with a list of captured operations:

```python
class Builder:
    _current = None

    def __init__(self):
        self.ops = []

    def __enter__(self):
        Builder._current = self
        return self

    def __exit__(self, exc_type, exc, tb):
        Builder._current = None
```

Tracing is then pleasantly direct:

```python
with Builder() as builder:
    fn(*symbolic_bound.args, **symbolic_bound.kwargs)

return builder.ops
```

The kernel function really runs. It just runs in a strange little symbolic
universe where addition creates `BinOp` and memory access creates `Load`.

This also means Python side effects would happen during tracing. A `print`
inside a kernel would print. Mutating a global list would mutate it. This is one
of the trade-offs of tracing, and something the language will eventually need
to diagnose or restrict more carefully.

## A symbolic value must not become a Python Boolean

This was an easy bug to miss.

Python considers ordinary objects truthy, so without extra protection this code
would silently choose one branch:

```python
if offsets < n:
    ...
```

But `offsets < n` is a vector comparison that should happen at runtime. It is
not one Boolean the Python interpreter can decide while tracing.

The safest behavior is to reject it:

```python
def __bool__(self) -> bool:
    raise TypeError(
        "Python control flow over symbolic values is not supported"
    )
```

I would much rather have a small compiler say "I cannot do that yet" than
quietly produce the wrong program. Proper control flow can come later with
proper IR support.

## Runtime values and compile-time values

Most kernel arguments become symbolic runtime parameters. Arrays become
`ptr<f32>`, integers become `i32`, and floating-point values become `f32`.

`BLOCK` is different:

```python
def add_kernel(..., BLOCK: tl.constexpr):
```

It controls the shape of `tl.arange`, so its value must be known while the
kernel is being traced. The tracer passes it through as a real Python integer.
As a result, `BLOCK=256` becomes `Const(256)` in the captured expression.

The launch wrapper also collects compile-time arguments into metadata for the
grid function:

```python
meta = {
    name: bound.arguments[name]
    for name, parameter in self.signature.parameters.items()
    if parameter.annotation is constexpr
}
```

There is a mildly annoying modern-Python detail here. With
`from __future__ import annotations`, annotations may be stored as strings. The
compiler resolves them when it wraps the function:

```python
annotations = inspect.get_annotations(fn, eval_str=True)
```

Without that step, `"tl.constexpr"` would not compare equal to the actual
`constexpr` marker, and `BLOCK` would accidentally become a runtime parameter.

## Recreating the launch syntax

The `@jit` decorator returns a `CompiledKernel` object:

```python
def jit(fn):
    return CompiledKernel(fn)
```

Its `__getitem__` method is what makes this syntax possible:

```python
add_kernel[grid](...)
```

At launch time, mytriton uses `inspect.Signature.bind` instead of manually
matching arguments:

```python
bound = self.signature.bind(*args, **kwargs)
bound.apply_defaults()
```

That lets Python handle positional arguments, keyword arguments, missing
arguments, and defaults consistently.

The grid function receives the compile-time metadata, returns dimensions, and
the wrapper validates that every dimension is a positive integer. Version 1
does not execute anything, so the grid is not used to launch programs yet. For
now it establishes the public API and makes the compile-time/runtime split
visible.

## What did we actually capture?

The full dataclass output is long, but the important shape is this:

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

with:

```text
offsets = program_id(0) * 256 + arange(0, 256)
mask    = offsets < n
```

That is enough to show that the frontend understood the important parts of the
kernel: parameters, constants, vector offsets, masks, pointer arithmetic,
loads, arithmetic, and the final store.

It is also enough to write a fairly strict test. The test traces the kernel and
compares the resulting dataclasses with the exact expected tree. It additionally
checks that the grid received:

```python
{"BLOCK": 256}
```

The test uses NumPy arrays and runs without a GPU. At this stage the arrays are
only there to provide dtype and layout information.

## What's next

The next big step is a typed SSA IR. Instead of one deeply nested `Store`, I want
something closer to:

```text
%pid = program_id 0
%range = arange 0, 256
%base = mul %pid, 256
%offsets = add %base, %range
%mask = cmp.lt %offsets, %n
%x_values = load %x, %offsets, %mask
%y_values = load %y, %offsets, %mask
%sum = add %x_values, %y_values
store %out, %offsets, %sum, %mask
```

Named results and explicit operations will make type checking, verification,
printing, constant folding, and dead-code elimination much easier. After that,
a tiny CPU interpreter would give the project an end-to-end execution path
without making CUDA the first dependency.

All code for this milestone is available at
[https://github.com/pbelevich/mytriton/tree/ver1](https://github.com/pbelevich/mytriton/tree/ver1).

Next: [Part 2]({% post_url 2026-06-23-My_Triton_Language_From_Scratch_-_Part_2 %}).
