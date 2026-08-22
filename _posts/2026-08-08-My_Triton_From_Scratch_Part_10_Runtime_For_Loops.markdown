---
layout: post
title:  "My Triton From Scratch Part 10: Runtime For Loops"
date:   2026-08-08 16:00:00 +0000
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
the middle of the compiler became stricter: SSA is verified, optimized, and
verified again before code generation.

In [Part 6: Reductions]({% post_url 2026-06-27-My_Triton_From_Scratch_Part_6_Reductions %}),
vectors learned how to cooperate inside one CUDA block: row-wise sum, max, min,
softmax, and a first naive matmul.

In [Part 7: Minimal MLIR]({% post_url 2026-06-28-My_Triton_From_Scratch_Part_7_Minimal_MLIR %}),
the compiler learned that optimized SSA can feed more than one backend.

In [Part 8: Rank-2 Tiles]({% post_url 2026-06-29-My_Triton_From_Scratch_Part_8_Rank_2_Tiles %}),
distributed values gained logical rows and columns.

In [Part 9: AST Frontend]({% post_url 2026-08-01-My_Triton_From_Scratch_Part_9_AST_Frontend %}),
Python stopped being the hidden frontend. mytriton began parsing the kernel
source and interpreting a supported subset of its AST itself.

Version 10 uses that frontend for the feature that motivated it: a real runtime
`for` loop.

## Why unrolling is not enough

The rank-2 matmul from Part 8 walks across the reduction dimension `K`:

```python
for k in tl.static_range(0, K):
    a_values = tl.load(a + offs_m * K + k)
    b_values = tl.load(b + k * N + offs_n)
    acc += a_values * b_values
```

`K` was a `tl.constexpr`, so the frontend could execute the loop while tracing.
If `K == 128`, the resulting IR contains 128 copies of the body. That is useful
for small fixed loops, but it has two obvious costs:

- generated SSA and CUDA source grow with `K`;
- the kernel must be recompiled for every new `K`.

For a runtime `K`, the kernel should instead contain a loop:

```python
for k in range(K):
    ...
```

This looks like a tiny source change. It is not a tiny compiler change. A loop
is the first construct in mytriton whose body is a nested region and whose
values cross a control-flow boundary.

Until now, SSA was a flat list:

```text
operation
operation
operation
store
```

Now an item in that list may itself contain a list of operations:

```text
operation
for (...) {
  operation
  operation
  yield
}
store
```

That affects tracing, type inference, SSA lowering, verification, printing,
CUDA code generation, optimization, and every analysis that walks the IR.

## Two meanings of `range`

The AST frontend evaluates the three components of a range first:

```python
start, stop, step = self._eval_range_parts(node.iter)
```

It then chooses between two paths.

If all three are exact Python integers, the loop is compile-time and is
unrolled by the frontend:

```python
if type(start) is int and type(stop) is int and type(step) is int:
    for value in range(start, stop, step):
        self.env[name] = value
        self.visit_stmt_list(node.body)
```

If at least one bound is symbolic, the frontend constructs a runtime loop:

```python
self._visit_runtime_for(node, start=start, stop=stop, step=step)
```

This keeps ordinary Python syntax while making the distinction explicit in the
frontend:

```python
for i in range(4):       # compile-time unrolled
    ...

for i in range(n):       # runtime SSA region
    ...
```

`tl.static_range` remains the way to demand compile-time iteration. Its bounds
must be integers. Runtime `range` in this milestone has deliberately narrow
semantics:

- the induction variable must be a simple name;
- `step` must be a positive compile-time integer;
- assignment to the induction variable is rejected;
- `for/else` is rejected;
- `break` and `continue` are not supported.

These restrictions leave one canonical loop shape for the first lowering:

```c++
for (int i = start; i < stop; i += step) {
    ...
}
```

## A loop is an expression-tree region

The expression-tree IR gets four new node kinds:

```python
@dataclass
class LoopIndex:
    name: str


@dataclass
class LoopCarry:
    index: int
    initial: Expression


@dataclass
class LoopResult:
    loop: ForRange
    index: int


@dataclass
class ForRange:
    index: LoopIndex
    start: Expression
    stop: Expression
    step: Expression
    captures: tuple[Expression, ...]
    body: list[TopLevelOp]
    carried_inputs: tuple[Expression, ...]
    carried_args: tuple[LoopCarry, ...]
    carried_outputs: tuple[Expression, ...]
    results: tuple[LoopResult, ...] = ()
```

The fields divide values by the way they interact with the region:

- `index` is defined by the loop itself;
- `captures` are outer values read in the body but not changed by it;
- `carried_inputs` initialize variables that change across iterations;
- `carried_args` are those variables as seen at the start of one iteration;
- `carried_outputs` are their values at the end of that iteration;
- `results` are the final values available after the loop.

The distinction between a capture and a carried value is important.

Consider:

```python
offsets = tl.arange(0, BLOCK)
acc = 0.0

for i in range(n):
    acc = acc + tl.load(x + offsets + i)
```

`offsets` is a capture. The loop reads the same tile on every iteration, and no
new version of `offsets` leaves the body.

`acc` is carried. Iteration zero sees the initial `0.0`; iteration one sees the
value produced by iteration zero; and the value produced by the final iteration
is visible after the loop.

## Finding loop-carried names

Python assignments are not SSA. Before tracing the body, the frontend must find
which source-level names may change.

An `AssignedNameCollector` walks the body and records simple assignment targets
in source order:

```python
class AssignedNameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: list[str] = []
        self._seen: set[str] = set()

    def add_name(self, name: str) -> None:
        if name not in self._seen:
            self._seen.add(name)
            self.names.append(name)
```

It recognizes normal, annotated, and augmented assignments. Only a name that
already existed before the loop becomes loop-carried:

```python
assigned = _assigned_names(node.body)
carried_names = tuple(name for name in assigned if name in self.env)
```

This rule handles the common accumulator pattern:

```python
acc = initial
for i in range(n):
    acc += update(i)
use(acc)
```

A temporary first created inside the body does not escape:

```python
for i in range(n):
    temporary = load(i)

# temporary is not defined here
```

Preserving source order is not only cosmetic. If a loop carries two variables,
their `iter_args`, `yield` operands, and results must use one stable order all
the way through the compiler.

## Tracing the body in an isolated builder

The body is traced with its own environment and its own `Builder`:

```python
loop_index = LoopIndex(target.id)
carried_inputs = tuple(unwrap(self.env[name]) for name in carried_names)
carried_args = tuple(
    LoopCarry(index=i, initial=initial)
    for i, initial in enumerate(carried_inputs)
)

body_env = dict(self.env)
body_env[target.id] = Value(loop_index)

for name, carried_arg in zip(carried_names, carried_args, strict=True):
    body_env[name] = Value(carried_arg)

with Builder() as body_builder:
    body_tracer = ASTTracer(body_env, self.external_env, capture_env=capture_env)
    body_tracer.visit_stmt_list(node.body)
```

Replacing `acc` with a `LoopCarry` in `body_env` is the expression-tree version
of a region block argument. Operations in the body do not refer directly to the
pre-loop `acc`; they refer to the value supplied to the current iteration.

After tracing, the environment contains the expressions produced by the body:

```python
carried_outputs = tuple(unwrap(body_env[name]) for name in carried_names)
```

Finally, each carried source name is rebound to a `LoopResult`:

```python
for name, result in zip(carried_names, results, strict=True):
    self.env[name] = Value(result)
```

Code after the loop therefore consumes the final loop result, not an expression
defined only inside the body.

## Why captures must be explicit

Expression lowering is lazy. A symbolic expression is emitted when an
observable operation, such as a store, needs it. Regions add a scope boundary
to that otherwise convenient model.

In the runtime matmul, `offs_m` and `offs_n` are created before the loop, read
inside it, and used again to form the final store:

```python
offs_m = ...
offs_n = ...
acc = ...

for k in range(K):
    a_offsets = offs_m * K + k
    b_offsets = k * N + offs_n
    acc += ...

c_offsets = offs_m * N + offs_n
tl.store(c + c_offsets, acc, ...)
```

Without explicit captures, lowering could first encounter the expressions for
`offs_m` and `offs_n` while lowering the nested body. Their SSA definitions
would then be placed inside the loop, even though the final store also uses
them outside it. The verifier would correctly report a use before definition
or a region value escaping its scope.

The body tracer records outer symbolic objects by identity, in first-use order.
SSA lowering forces those captures into the enclosing region before it switches
to the body operation list:

```python
start = self.lower_expr(loop.start)
stop = self.lower_expr(loop.stop)
step = self.lower_expr(loop.step)

for capture in loop.captures:
    self.lower_expr(capture)

carried_inputs = tuple(
    self.lower_expr(value) for value in loop.carried_inputs
)

self.ops = []
# lower the nested body here
```

Captures are not additional CUDA loop parameters. They are SSA values defined
in the enclosing scope and used by the nested region, just as a C++ loop body
can read a local declared before the loop.

## Structured SSA

A flat `SSAOp` is no longer enough, so the SSA item type becomes a union:

```python
@dataclass
class SSAForRange:
    index: SSAValue
    start: SSAOperand
    stop: SSAOperand
    step: SSAOperand
    carried_inputs: tuple[SSAOperand, ...]
    carried_args: tuple[SSAValue, ...]
    body: list["SSAItem"]
    yields: tuple[SSAOperand, ...]
    results: tuple[SSAValue, ...]


SSAItem = SSAOp | SSAForRange
```

`body` contains `SSAItem`, not only `SSAOp`. This recursive definition is what
allows nested loops.

Here is the central part of the runtime matmul SSA for a `4 x 8` output tile:

```text
%12 = mul %11, 0.0 : block<4x8 x f32>
%27 = for %13 in range(0, K, 1) iter_args(%14 = %12) : block<4x8 x f32> {
  %15 = mul %4, K : block<4x1 x i32>
  %16 = add %15, %13 : block<4x1 x i32>
  %17 = addptr a, %16 : block<4x1 x ptr<f32>>
  %18 = cmp_lt %4, M : block<4x1 x bool>
  %19 = load %17, %18, 0.0 : block<4x1 x f32>
  %20 = mul %13, N : i32
  %21 = add %20, %9 : block<1x8 x i32>
  %22 = addptr b, %21 : block<1x8 x ptr<f32>>
  %23 = cmp_lt %9, N : block<1x8 x bool>
  %24 = load %22, %23, 0.0 : block<1x8 x f32>
  %25 = mul %19, %24 : block<4x8 x f32>
  %26 = add %14, %25 : block<4x8 x f32>
  yield %26
}
```

This syntax is modeled after region-based SSA IRs, but it is still mytriton's
small textual format.

Read the loop header from left to right:

```text
%27 = for %13 in range(0, K, 1) iter_args(%14 = %12) : block<4x8 x f32> {
```

- `%13` is the induction variable inside the body;
- the loop runs from `0` while `%13 < K`, in steps of `1`;
- `%12` is the accumulator before the first iteration;
- `%14` is the accumulator at the start of the current iteration;
- `%27` is the final accumulator after the loop;
- all three accumulator values have type `block<4x8 x f32>`.

The terminator:

```text
yield %26
```

does **not** mean Python `yield`, and it does not exit the loop early. It means:

1. use `%26` as `%14` at the start of the next iteration;
2. if this was the final iteration, use `%26` as the loop result `%27`.

For zero iterations, `%27` is the initial `%12`.

With multiple carried variables the structure simply has parallel lists:

```text
%8, %9 = for %2 in range(0, n, 1)
    iter_args(%3 = %0, %4 = %1) : i32, f32 {
  ...
  yield %6, %7
}
```

Argument zero is always paired with yield zero and result zero, which is why
stable carried-name ordering matters.

## Nested loops and indentation

Because the body is recursive, the printer is recursive too:

```python
for body_op in loop.body:
    body_lines = (
        self.print_for_range(body_op)
        if isinstance(body_op, SSAForRange)
        else [self.print_op(body_op)]
    )
    lines.extend(f"  {line}" for line in body_lines)
```

Each enclosing loop adds two spaces to every line returned for its body. A
nested loop therefore prints naturally:

```text
%7 = for %1 in range(0, M, 1) iter_args(%2 = 0) : i32 {
  %6 = for %3 in range(0, N, 1) iter_args(%4 = %2) : i32 {
    %5 = add %4, 1 : i32
    yield %5
  }
  yield %6
}
```

The indentation is only presentation. The semantic nesting comes from
`SSAForRange.body` containing another `SSAForRange` object.

## Type inference across the region

The new expression nodes have simple but essential types:

- a `LoopIndex` is `i32`;
- a `LoopCarry` has the type of its initial value;
- a `LoopResult` has the type of the corresponding carried output.

The loop does not perform an implicit promotion between iterations. If an
`i32` carried argument is yielded as `f32`, verification fails. The initial
value, region argument, yielded value, and final result must all agree.

This invariant makes loop lowering predictable for every backend. There is no
hidden mutable Python variable after the AST stage; there is a typed cycle
represented by explicit region arguments and results.

## Verification becomes recursive

The verifier now carries a set of definitions into each region. It checks that:

- loop bounds are scalar `i32` values;
- the constant step is positive;
- induction and carried-argument IDs are fresh;
- the body uses only enclosing definitions or definitions earlier in the body;
- body-local definitions do not escape the region;
- carried inputs, carried arguments, yields, and results have equal counts;
- the types in each carried-value lane match;
- nested loops obey the same rules recursively.

That region scope catches real compiler bugs. An SSA value created inside the
body cannot be used after the closing brace unless the loop explicitly yields
it as a result.

This is also why captures are lowered before the body. Verification should not
special-case a broken definition order; the frontend and lowering must produce
well-scoped SSA.

## CUDA lowering

CUDA C++ has mutable locals, so the structured SSA maps to a conventional loop.
The carried result is declared and initialized before the loop:

```c++
float v27 = v12;
```

Both the region argument `%14` and the result `%27` map to this same C++ local.
The loop body computes its yielded value and assigns it back at the end of the
iteration:

```c++
for (int v13 = 0; v13 < K; v13 += 1) {
    int v15 = (v4 * K);
    int v16 = (v15 + v13);
    bool v18 = (v4 < M);
    float v19 = (v18 ? a[v16] : 0.0f);
    int v20 = (v13 * N);
    int v21 = (v20 + v9);
    bool v23 = (v9 < N);
    float v24 = (v23 ? b[v21] : 0.0f);
    float v25 = (v19 * v24);
    float v26 = (v27 + v25);
    v27 = v26;
}
```

The assignment `v27 = v26` is the CUDA rendering of `yield %26`.

Nested CUDA loops use the same recursive emission strategy as the SSA printer.
The code generator emits a nested body, then adds one indentation level to all
of its generated lines.

## Analyses must visit regions

Adding a structured operation exposes a useful compiler-design rule: an IR
walker that only knows about a flat list is now incomplete.

Version 10 updates shape analysis and verification to recurse into loop bodies.
The CUDA backend does the same during emission.

The existing optimization passes are intentionally more conservative. Common
subexpression elimination and dead-code elimination were written for a flat
sequence, so kernels containing runtime loops skip those rewrites for now.
Silently applying a non-region-aware optimizer would be worse than missing an
optimization: it could move, remove, or merge values across invalid scopes.

The minimal MLIR backend also rejects runtime loops explicitly. Lowering an
`SSAForRange` to `scf.for` will be a separate milestone; Version 10 only adds
the CUDA path.

## The runtime matmul kernel

The complete loop shape is now the one we wanted in Part 9:

```python
@triton.jit
def matmul_runtime_k_kernel(
    a,
    b,
    c,
    M,
    N,
    K,
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

    for k in range(K):
        a_offsets = offs_m * K + k
        b_offsets = k * N + offs_n
        a_values = tl.load(a + a_offsets, mask=offs_m < M, other=0.0)
        b_values = tl.load(b + b_offsets, mask=offs_n < N, other=0.0)
        acc = acc + a_values * b_values

    tl.store(c + c_offsets, acc, mask=c_mask)
```

`K` is now a normal runtime `i32` parameter. The JIT cache no longer needs a
separate specialization for every value of `K`, and generated code contains
one body rather than `K` unrolled copies.

The accumulator initialization is still awkward:

```python
acc = c_offsets * 0.0
```

It exists only because multiplying the integer `BM x BN` offset tile by a
floating-point zero creates a `BM x BN` floating-point tile. The kernel
language still has no direct way to say “make a zero tile with this shape and
dtype.” That is the next gap to close.

## What changed conceptually

Before Version 10, repeated work existed only by duplicating expressions during
tracing:

```text
Python loop -> many flat SSA operations
```

Now repetition can survive as a compiler object:

```text
Python runtime loop
        |
        v
expression-tree ForRange
        |
        v
typed SSA region with iter_args and yield
        |
        v
CUDA C++ for loop
```

The important addition is not the C++ `for` syntax. It is the explicit
control-flow boundary in the middle of the compiler, with well-defined inputs,
captures, region arguments, yielded values, and results.

That structure is enough for runtime and nested loops today. Later it can also
support region-aware optimization and another structured backend lowering.

All code for this milestone is available at
[https://github.com/pbelevich/mytriton/tree/ver10](https://github.com/pbelevich/mytriton/tree/ver10).

Next: [Part 11: Block Factory Functions]({% post_url 2026-08-15-My_Triton_From_Scratch_Part_11_Block_Factory_Functions %}).
