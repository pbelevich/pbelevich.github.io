---
layout: post
title:  "My Triton From Scratch Part 9: AST Frontend"
date:   2026-08-01 16:00:00 +0000
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

In [Part 8: Rank-2 Tiles]({% post_url 2026-06-29-My_Triton_From_Scratch_Part_8_Rank_2_Tiles %}),
distributed values gained logical rows and columns instead of pretending every
block was one flat vector.

Version 9 comes back to the very beginning of the pipeline.

The practical motivation is matrix multiplication. A matmul kernel needs to
walk over its inner dimension `K`, load one slice from `A` and one slice from
`B`, and accumulate their product into an output tile:

```python
for k in tl.static_range(0, K):
    a_values = tl.load(a + offs_m * K + k)
    b_values = tl.load(b + k * N + offs_n)
    acc += a_values * b_values
```

I want that loop to be part of the kernel language, with semantics controlled
by mytriton. Relying on CPython to execute it happens to unroll a constexpr
`K`, but it leaves the compiler unable to see that the source contained a loop.
An AST frontend gives the compiler an explicit place to recognize iteration
over the inner matmul dimension today and to lower richer loop forms later.

Until now, tracing meant calling the kernel as an ordinary Python function with
symbolic arguments:

```python
with Builder() as builder:
    fn(*symbolic_args, **symbolic_kwargs)
```

This technique carried mytriton surprisingly far. Python executed assignments,
function calls, operators, indexing, and compile-time loops. The symbolic
objects only had to overload operations such as `+`, `<`, `&`, and `[]` to
build the expression tree.

But that also meant Python itself was the frontend.

mytriton could see the operations that happened while the function ran, but it
could not see the syntax that caused them. A `for` loop was already gone by the
time symbolic values reached the builder. An assignment was only a Python local
variable update. Unsupported syntax failed wherever the Python interpreter or
one of the symbolic objects happened to notice it.

Version 9 inserts an explicit frontend before symbolic tracing:

```text
Python source
    -> Python AST
    -> AST frontend
    -> expression-tree IR
    -> typed SSA
    -> verification and optimization
    -> CUDA C++ or MLIR
```

The generated expression tree, SSA, and backend source stay the same. The new
layer changes who is responsible for understanding the kernel source.

The code for this milestone is here:
[https://github.com/pbelevich/mytriton/tree/ver9](https://github.com/pbelevich/mytriton/tree/ver9).

## The kernel still looks the same

The AST frontend is not introduced with a new kernel operation. Existing
kernels are the test.

For example, the rank-2 matmul from Part 8 still contains assignments,
attributes, calls, indexing, a compile-time loop, augmented assignment, and a
masked store:

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

        acc += a_values * b_values

    tl.store(c + c_offsets, acc, mask=c_mask)
```

Earlier versions asked CPython to execute this function. Version 9 asks the
standard-library [`ast`](https://docs.python.org/3/library/ast.html) module to
parse it, then evaluates the supported nodes deliberately.

That is an important constraint: this is not a second Python interpreter. It is
a small language frontend whose source syntax happens to be a carefully chosen
subset of Python.

## Finding the function in its source

The frontend begins with the original function object. Python function objects
contain code objects, globals, defaults, annotations, and closure information,
but they do not directly contain a high-level syntax tree.

So the first step recovers and parses the source:

```python
def _find_function_def(fn) -> ast.FunctionDef:
    source = inspect.getsource(fn)
    tree = ast.parse(textwrap.dedent(source))

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn.__name__:
            return node

    raise ASTFrontendError(
        f"could not find function definition for {fn.__name__}"
    )
```

`inspect.getsource` returns the source lines that define the function.
`textwrap.dedent` matters for kernels nested inside tests or helper functions:
their source is indented in the file, but `ast.parse` expects the snippet to be
valid at its new top level.

The result is an `ast.FunctionDef`. Its `body` is an ordered list of statement
nodes. The frontend does not visit the decorator or reconstruct a Python
call. It starts directly with those statements.

This MVP therefore needs source to be available. A function created with
`eval`, recovered only from bytecode, or defined in an environment where
`inspect.getsource` cannot find its file is outside the current boundary.

That is a reasonable boundary for a source-language frontend. More importantly,
it is now an explicit one.

## Two environments, two kinds of names

Evaluating a kernel AST requires resolving names such as:

```python
a
K
acc
tl
range
```

Those names do not all come from the same place.

Kernel parameters and local assignments live in the frontend's symbolic
environment:

```python
self.env
```

Globals, nonlocals, and builtins live in an external environment:

```python
self.external_env
```

The initial symbolic environment preserves the runtime/constexpr split from
the old tracer:

```python
def _make_symbolic_env(signature, bound_args, runtime_params):
    env = {}
    params_by_name = {param.name: param for param in runtime_params}

    for name, parameter in signature.parameters.items():
        value = bound_args[name]

        if is_constexpr_annotation(parameter.annotation):
            env[name] = value
            continue

        param = params_by_name[name]
        if isinstance(param.ty, PointerType):
            env[name] = Ptr(param)
        else:
            env[name] = Value(param)

    return env
```

A compile-time parameter such as `K: tl.constexpr` stays an ordinary Python
integer. A pointer parameter becomes `Ptr`. Another runtime scalar becomes
`Value`.

That distinction is why this loop can be unrolled:

```python
for k in tl.static_range(0, K):
```

`K` is already a concrete integer in the frontend environment.

The external environment comes from the function object:

```python
closure_vars = inspect.getclosurevars(fn)
external_env = {
    **closure_vars.builtins,
    **closure_vars.globals,
    **closure_vars.nonlocals,
}
```

This is how `tl`, a module-level helper, a captured nonlocal, or builtin
`range` can be found without pretending they are kernel parameters.

Name lookup checks locals first:

```python
def visit_Name(self, node):
    if isinstance(node.ctx, ast.Load):
        if node.id in self.env:
            return self.env[node.id]

        if node.id in self.external_env:
            return self.external_env[node.id]

        raise NameError(node.id)
```

That matches the useful part of Python's name-shadowing behavior while keeping
the frontend implementation small.

## Statements update the frontend environment

The AST visitor processes a function body in source order:

```python
def visit_stmt_list(self, body):
    for stmt in body:
        self.visit(stmt)
```

A simple assignment evaluates its right-hand side and records the result:

```python
def visit_Assign(self, node):
    if len(node.targets) != 1:
        raise ASTFrontendError(
            "only single-target assignment is supported"
        )

    target = node.targets[0]
    if not isinstance(target, ast.Name):
        raise ASTFrontendError(
            "only assignment to a simple name is supported"
        )

    self.env[target.id] = self.visit(node.value)
```

For:

```python
offsets = pid * BLOCK + tl.arange(0, BLOCK)
```

the right-hand side produces a symbolic `Value`, and `offsets` becomes a name
for that value in `self.env`.

Assignment itself does not create a new expression-tree operation. That is the
same semantic result as before: Python local names are handles for symbolic
expressions, not mutable storage in the kernel IR.

Annotated assignment is also accepted:

```python
value: something = expression
```

The MVP records the value but does not interpret or enforce the local
annotation. An annotation without a value creates a local bound to `None`.

Augmented assignment is made explicit too:

```python
acc += a_values * b_values
```

becomes conceptually:

```python
self.env["acc"] = self.env["acc"] + rhs
```

The frontend supports `+=`, `-=`, `*=`, and `/=` on simple names. Attribute,
subscript, tuple-unpacking, and chained assignment targets are rejected instead
of acquiring accidental semantics.

Expression statements are evaluated and their return value is ignored. That is
exactly what a side-effecting call such as `tl.store(...)` needs: the call adds
a `Store` node to the active builder even though there is no local result to
remember.

## Expressions reuse the symbolic language

The AST frontend does not rebuild the expression-tree IR itself. It evaluates
syntax into the same `Value`, `Ptr`, and language functions used by the old
tracer.

For a binary operation, it first evaluates both operands:

```python
lhs = self.visit(node.left)
rhs = self.visit(node.right)
```

Then it applies the corresponding Python operator:

```python
if isinstance(node.op, ast.Add):
    return lhs + rhs

if isinstance(node.op, ast.Sub):
    return lhs - rhs

if isinstance(node.op, ast.Mult):
    return lhs * rhs

if isinstance(node.op, ast.Div):
    return lhs / rhs

if isinstance(node.op, ast.BitAnd):
    return lhs & rhs
```

If `lhs` or `rhs` is symbolic, its overload creates the same `BinOp` or
`AddPtr` node as before. If both are constexpr values, Python computes an
ordinary compile-time result.

This is a useful division of responsibility:

```text
AST frontend: what syntax did the user write?
symbolic values: what expression-tree node does that operation mean?
```

Unary `+` and `-`, simple `<` and `is` comparisons, tuples, lists, and constants
follow the same pattern.

Calls are similarly direct:

```python
def visit_Call(self, node):
    fn = self.visit(node.func)
    args = [self.visit(arg) for arg in node.args]
    kwargs = {
        keyword.arg: self.visit(keyword.value)
        for keyword in node.keywords
    }
    return fn(*args, **kwargs)
```

For `tl.load(...)`, `visit_Attribute` first resolves `tl`, then `getattr`
resolves `load`. The evaluated function receives symbolic pointers, masks, and
values and builds a normal `Load` expression.

`**kwargs` are deliberately unsupported. The frontend wants each supported
source construct to be visible in its own implementation instead of delegating
arbitrary Python call behavior back to CPython.

## Indexing is narrow on purpose

Part 8 added exactly two symbolic indexing forms:

```python
x[:, None]
x[None, :]
```

The AST for these expressions contains `Subscript`, `Tuple`, `Slice`, and
`Constant(None)` nodes. The frontend turns them back into ordinary Python index
objects:

```python
def _eval_index(self, node):
    if isinstance(node, ast.Tuple):
        return tuple(self._eval_index(elt) for elt in node.elts)

    if isinstance(node, ast.Slice):
        if node.lower is None and node.upper is None and node.step is None:
            return slice(None)
        raise ASTFrontendError("only ':' slices are supported")

    if isinstance(node, ast.Constant) and node.value is None:
        return None

    return self.visit(node)
```

Then `visit_Subscript` applies that index:

```python
value = self.visit(node.value)
index = self._eval_index(node.slice)
return value[index]
```

The symbolic `Value.__getitem__` from Part 8 still decides that `[:, None]`
means `ExpandDims(axis=1)` and `[None, :]` means `ExpandDims(axis=0)`.

This is another useful separation. The frontend understands Python's spelling
of an index. The symbolic language understands which IR operation that index
means.

General slicing, computed slice bounds, ellipsis, and arbitrary symbolic
indexing remain unsupported.

## Compile-time loops are interpreted, not captured

The most compiler-shaped statement in Version 9 is `for`.

It is also the main reason for switching frontends. The inner dimension of
matmul is naturally expressed as a loop over `K`. Version 9 still unrolls that
loop because `K` is constexpr, but the decision is now made by mytriton after
it recognizes an `ast.For`, rather than being an invisible side effect of
calling the Python function.

The MVP accepts only a simple induction variable and one of two iterators:

```python
for k in range(...):
```

or:

```python
for k in tl.static_range(...):
```

The range can use one, two, or three positional arguments. Every bound and the
step must evaluate to an exact Python `int`:

```python
for name, value in (
    ("start", start),
    ("stop", stop),
    ("step", step),
):
    if type(value) is not int:
        raise ASTFrontendError(
            "dynamic range bounds are not supported by "
            f"AST frontend MVP; {name} is {value!r}"
        )
```

The frontend then performs ordinary compile-time unrolling:

```python
for value in loop_range:
    self.env[name] = value
    self.visit_stmt_list(node.body)
```

For `K = 3`, the matmul body is visited three times. Each visit adds another
set of loads, multiplication, and accumulator update to the expression tree.
There is still no loop operation in SSA or generated CUDA.

This matches the behavior of the previous `tl.static_range` helper, which
returned a Python `range`. The difference is that unrolling now belongs to the
frontend and can eventually evolve independently from Python execution.

The induction variable is scoped carefully. If the same name existed before
the loop, its old value is restored. Otherwise the temporary name is removed
after unrolling.

The current frontend rejects `for/else`, non-name targets, arbitrary iterables,
keyword arguments to ranges, and dynamic bounds. Those constructs would need
new language semantics, not just more visitor methods.

## Compile-time conditional expressions

Version 9 supports Python's conditional expression:

```python
value = lhs if CONDITION else rhs
```

but only when `CONDITION` evaluates to an ordinary Python Boolean:

```python
def visit_IfExp(self, node):
    test = self.visit(node.test)
    if not isinstance(test, bool):
        raise ASTFrontendError(
            "only constexpr conditions in IfExp are supported"
        )
    return self.visit(node.body) if test else self.visit(node.orelse)
```

Only the selected branch is visited. This is compile-time specialization, not
runtime control flow.

A symbolic condition cannot silently become Python control flow. The frontend
rejects it because representing that condition would require branch operations,
blocks, and merge semantics in the IR.

There is also no statement-form `if` yet. Supporting a constexpr `if` statement
would be a small extension. Supporting a runtime symbolic `if` would be a much
larger compiler milestone.

## Unsupported Python becomes a frontend error

`ast.NodeVisitor` normally walks unknown nodes recursively. That default is
convenient for analysis tools, but dangerous for a language frontend: a new
syntax form could appear to work while important semantics are skipped.

Version 9 reverses the default:

```python
def generic_visit(self, node):
    raise ASTFrontendError(
        f"unsupported AST node: {type(node).__name__}"
    )
```

Every accepted syntax node therefore needs an explicit visitor method.

Examples of honest failures include:

```text
unsupported AST node: If
unsupported AST node: While
unsupported AST node: BoolOp
return statements are not supported in kernels
only assignment to a simple name is supported
dynamic range bounds are not supported by AST frontend MVP
```

The exact supported subset is small:

- expression, assignment, annotated assignment, and augmented assignment
  statements;
- compile-time `for` loops over `range` or `tl.static_range`;
- names, constants, tuples, lists, attributes, calls, and narrow subscripts;
- `+`, `-`, `*`, `/`, and `&`;
- unary `+` and `-`;
- simple `<` and `is` comparisons;
- constexpr conditional expressions.

That list is not an attempt to describe all of Python. It is a language
contract for the kernels mytriton can currently compile.

This also improves precedence diagnostics. A chained comparison is rejected
with a reminder to parenthesize both sides of `&`:

```text
only simple comparisons are supported; when combining comparisons with &,
wrap each comparison in parentheses
```

The frontend sees the malformed syntax shape directly. It no longer has to
wait for symbolic `__bool__` to fail somewhere during Python execution.

## The rest of the compiler does not change

The public compilation path changes in one important import:

```python
from .ast_frontend import trace
```

instead of:

```python
from .trace import trace
```

The old `trace` function in `trace.py` is removed. Its symbolic classes,
expression nodes, builder, parameter construction, and language operations all
remain.

The new frontend ends with the same builder protocol:

```python
with Builder() as builder:
    ASTTracer(env, external_env).visit_stmt_list(function_def.body)

return builder.ops, runtime_params
```

So every later stage receives the same interface:

```text
list of expression-tree operations + runtime parameters
```

Type inference does not need to know whether CPython or `ASTTracer` produced
the tree. SSA lowering does not change. The verifier and optimizers do not
change. CUDA and MLIR code generation do not change.

That lack of downstream churn is part of the result. The expression-tree IR
was already a real boundary. Version 9 replaces the implementation on one side
of that boundary without rewriting the other side.

## Existing kernels become frontend tests

Version 9 mostly moves test imports from:

```python
from mytriton.trace import trace
```

to:

```python
from mytriton.ast_frontend import trace
```

The expected expression trees and SSA stay unchanged.

That means the existing kernel suite now exercises the AST frontend across a
fairly useful source subset:

- vector and matrix elementwise kernels cover assignments, calls, arithmetic,
  comparisons, masks, and stores;
- rank-2 tile kernels cover tuple subscripts, full slices, `None`, attributes,
  and `&`;
- reduction and softmax kernels cover nested symbolic expressions;
- naive and rank-2 matmul cover constexpr parameters, `tl.static_range`, loop
  induction variables, and repeated accumulator updates;
- long-row sum covers compile-time `range`-style unrolling.

This is a compatibility test for the new frontend architecture. If the AST
visitor gives an existing source construct different meaning, the exact tree,
SSA, or generated-source assertions expose it.

There are many dedicated negative frontend tests still worth adding: unknown
names, multiple assignment targets, `while`, dynamic ranges, `for/else`,
symbolic conditional expressions, unsupported slices, and return statements.
The explicit errors are already implemented; a focused test file would make
that language boundary even more visible.

## Current boundaries

Version 9 is intentionally not a full Python frontend.

It depends on `inspect.getsource`, so source must be recoverable for the kernel
function.

Only ordinary `def` functions are found. Async functions and lambdas are not
part of the kernel language.

Assignments target simple local names. Destructuring, attribute assignment,
subscript assignment, chained assignment, `break`, `continue`, exceptions,
context managers, and nested definitions are not supported.

Loops are compile-time only. There is no loop representation in expression IR
or SSA, and runtime scalar values cannot control iteration counts.

Conditional expressions are constexpr only. Statement `if` and runtime control
flow are not represented.

Calls still invoke resolved Python callables. That is useful for the small
`mytriton.language` API and compile-time helpers, but it is not yet a restricted
or sandboxed execution model. An AST frontend makes syntax explicit; it does
not automatically make arbitrary external calls safe or compilable.

The frontend uses Python values as its compile-time value system. There is no
separate constant evaluator yet.

These boundaries are real, but the failure mode is now much better. Unsupported
syntax is rejected at the frontend with a local explanation instead of being
partially executed and failing in an unrelated symbolic overload.

## What changed conceptually

Before Version 9, mytriton traced a kernel by doing this:

```text
run the Python function and observe symbolic operations
```

That is a powerful prototype technique. It made Version 1 tiny and let the
project focus first on expression nodes, types, SSA, verification, optimization,
and backends.

But it left one compiler layer implicit. Python decided how statements and
expressions executed, and mytriton only participated when symbolic objects were
involved.

For matmul, that meant Python owned the loop over the inner dimension. The
compiler saw only the operations left after unrolling and had no source-level
loop to inspect, validate, transform, or eventually lower without unrolling.

After Version 9, tracing means:

```text
parse the Python function and interpret a supported kernel-language AST
```

The symbolic value machinery still builds the expression tree. The important
change is that mytriton now owns the walk from source syntax to those symbolic
operations.

That creates a real place for the next frontend semantics:

- runtime matmul loops over the inner dimension that survive into IR instead of
  being fully unrolled;
- constexpr `if` statements;
- better source-position diagnostics;
- explicit compile-time evaluation;
- runtime control flow once the IR can represent blocks and branches;
- language-specific restrictions before arbitrary Python executes;
- transformations that need to see syntax before it disappears.

Version 9 does not implement those next steps yet. It makes compile-time loop
unrolling explicit and creates the layer where runtime loops and the other
features belong.

The conceptual shift is small enough to fit in one line:

```text
Python is now the syntax of the kernel language, not its hidden frontend.
```

All code for this milestone is available at
[https://github.com/pbelevich/mytriton/tree/ver9](https://github.com/pbelevich/mytriton/tree/ver9).

Next: [Part 10: Runtime For Loops]({% post_url 2026-08-08-My_Triton_From_Scratch_Part_10_Runtime_For_Loops %}).
