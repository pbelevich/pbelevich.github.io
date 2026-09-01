---
layout: post
title:  "My Triton From Scratch Part 13: tl.dot Semantics"
date:   2026-08-23 16:00:00 +0000
# categories:
---

In [Part 1: Symbolic Tracing]({% post_url 2026-06-22-My_Triton_From_Scratch_Part_1_Symbolic_Tracing %}),
mytriton learned how to build a symbolic expression tree from a Python kernel.

In [Part 2: Typed SSA]({% post_url 2026-06-23-My_Triton_From_Scratch_Part_2_Typed_SSA %}),
that tree gained explicit types and SSA values.

In [Part 3: CUDA Lowering]({% post_url 2026-06-24-My_Triton_From_Scratch_Part_3_CUDA_Lowering %}),
typed SSA became executable CUDA C++.

In [Part 4: Elementwise Ops]({% post_url 2026-06-25-My_Triton_From_Scratch_Part_4_Elementwise_Ops %}),
the language grew a useful elementwise vocabulary.

In [Part 5: Verification]({% post_url 2026-06-26-My_Triton_From_Scratch_Part_5_Verification %}),
the compiler began checking and optimizing its SSA contract.

In [Part 6: Reductions]({% post_url 2026-06-27-My_Triton_From_Scratch_Part_6_Reductions %}),
CUDA threads learned to cooperate through shared memory.

In [Part 7: Minimal MLIR]({% post_url 2026-06-28-My_Triton_From_Scratch_Part_7_Minimal_MLIR %}),
optimized SSA gained a second lowering path.

In [Part 8: Rank-2 Tiles]({% post_url 2026-06-29-My_Triton_From_Scratch_Part_8_Rank_2_Tiles %}),
logical blocks gained rows, columns, and broadcasting.

In [Part 9: AST Frontend]({% post_url 2026-08-01-My_Triton_From_Scratch_Part_9_AST_Frontend %}),
mytriton took ownership of Python kernel syntax.

In [Part 10: Runtime For Loops]({% post_url 2026-08-08-My_Triton_From_Scratch_Part_10_Runtime_For_Loops %}),
runtime iteration became a structured SSA region with carried values.

In [Part 11: Block Factory Functions]({% post_url 2026-08-15-My_Triton_From_Scratch_Part_11_Block_Factory_Functions %}),
kernels gained honest block-valued accumulators through `tl.zeros`.

In [Part 12: CUDA Tile Layouts]({% post_url 2026-08-22-My_Triton_From_Scratch_Part_12_CUDA_Tile_Layouts %}),
logical tile shapes were separated from their physical CUDA thread layouts.

The language can now spell every individual multiply and add in a matrix
multiplication. That is enough for a naive kernel, but it hides the operation
the compiler actually needs to understand:

```text
A [M, K] × B [K, N] -> C [M, N]
```

Version 13 introduces that operation as `tl.dot`. This milestone deliberately
defines its semantics without implementing CUDA execution. Before deciding how
to move tiles through shared memory or which instructions should compute the
product, the frontend and IR need an unambiguous answer to a simpler question:
what does `dot` mean?

## Why `dot` must be an operation

The naive matmul from earlier parts expresses one output element by loading one
column of `A`, one row of `B`, multiplying them elementwise, and reducing the
result. Repeating that construction for every output coordinate works, but the
compiler sees only generic operations:

```text
load -> mul -> sum
```

It cannot tell that several threads are computing one matrix product. That
means it has no stable place to attach shared-memory staging, accumulator
ownership, tensor-core rules, or future layout transformations.

An explicit operation preserves the intent:

```text
lhs [M, K] ----\
                 dot ----> result [M, N]
rhs [K, N] ----/
```

The same language operation can eventually have several lowerings:

```text
tl.dot
  |
  +-- ordinary CUDA-core FMA loop
  +-- tensor-core mma instructions
  +-- MLIR dot-like operation
```

Version 13 only builds the common semantic part above those choices.

## The public API stays small

The language entry point is intentionally unsurprising:

```python
def dot(lhs: Value, rhs: Value) -> Value:
    return Value(Dot(unwrap(lhs), unwrap(rhs)))
```

It accepts two symbolic values, unwraps them to expression-tree nodes, creates
a `Dot`, and returns another symbolic `Value`. A kernel can therefore write:

```python
lhs = tl.zeros((4, 16), tl.float32)
rhs = tl.zeros((16, 8), tl.float32)
result = tl.dot(lhs, rhs)
```

There is no CUDA-specific argument here. The public operation does not mention
threads, warps, shared buffers, or tensor-core fragments. Those are lowering
decisions, not source-language semantics.

## A dedicated expression-tree node

The trace IR gains a node with exactly two children:

```python
@dataclass
class Dot:
    lhs: Expression
    rhs: Expression
```

`Dot` also becomes a member of the `Expression` union. This small addition is
important: the operation remains visible while Python execution constructs the
symbolic graph.

For the example above, the relevant part of that graph is:

```text
Zeros(shape=(4, 16), dtype=f32) ----\
                                      Dot
Zeros(shape=(16, 8), dtype=f32) ----/
```

The node still has no type field. As elsewhere in mytriton, type inference is a
separate compiler phase.

## The shape contract

Matrix multiplication has stricter rules than ordinary block broadcasting.
For:

```text
lhs: block<MxK x f32>
rhs: block<KxN x f32>
```

the result is:

```text
block<MxN x f32>
```

The first implementation encodes that contract directly:

```python
def infer_dot(self, expr: Dot) -> BlockType:
    lhs_ty = self.infer(expr.lhs)
    rhs_ty = self.infer(expr.rhs)

    if not isinstance(lhs_ty, BlockType) or lhs_ty.rank != 2:
        raise TypeError(f"dot lhs must be a rank-2 block, got {lhs_ty}")

    if not isinstance(rhs_ty, BlockType) or rhs_ty.rank != 2:
        raise TypeError(f"dot rhs must be a rank-2 block, got {rhs_ty}")

    if lhs_ty.element != F32:
        raise TypeError(f"dot lhs must have f32 elements, got {lhs_ty}")

    if rhs_ty.element != F32:
        raise TypeError(f"dot rhs must have f32 elements, got {rhs_ty}")

    lhs_rows, lhs_columns = lhs_ty.shape
    rhs_rows, rhs_columns = rhs_ty.shape

    if lhs_columns != rhs_rows:
        raise TypeError(
            "dot inner dimensions must match, "
            f"got {lhs_ty.shape} and {rhs_ty.shape}"
        )

    return BlockType((lhs_rows, rhs_columns), F32)
```

There are four independent requirements:

1. both operands are blocks;
2. both blocks have rank two;
3. both element types are `f32`;
4. the two `K` dimensions match.

It is worth keeping these checks separate. A bad rank and a bad reduction
dimension are different source mistakes and should produce different compiler
diagnostics.

Degenerate matrix dimensions remain valid. For example:

```text
[1, 7] × [7, 1] -> [1, 1]
[3, 1] × [1, 5] -> [3, 5]
```

They are still matrices, not scalar or vector special cases.

## Why only `f32` for now

Real Triton supports a much richer dot type system. Inputs may be `fp16` or
`bf16`, accumulation may use `f32`, and hardware-specific precision modes may
change the chosen instruction.

Adding all of that now would combine two separate questions:

```text
What shape does matrix multiplication produce?
What numeric contract does mixed-precision multiplication use?
```

Version 13 answers the first with the smallest useful numeric surface: both
operands and the result use `f32`. Mixed precision remains an explicit future
milestone rather than an accidental collection of conversions.

## Lowering to typed SSA

The SSA lowering recursively lowers both operands and uses the common `emit`
helper to infer the result type and create an ordinary result-producing
operation:

```python
if isinstance(expr, Dot):
    lhs = self.lower_expr(expr.lhs)
    rhs = self.lower_expr(expr.rhs)

    return self.emit(
        "dot",
        expr,
        operands=(lhs, rhs),
    )
```

The earlier example prints as:

```text
%0 = zeros {shape=(4, 16), dtype=f32} : block<4x16 x f32>
%1 = zeros {shape=(16, 8), dtype=f32} : block<16x8 x f32>
%2 = dot %0, %1 : block<4x8 x f32>
```

This line is the lasting contract of the version:

```text
%2 = dot %0, %1 : block<4x8 x f32>
```

Later backends can replace it with loops, shared-memory operations, or tensor
instructions, but they all start from the same typed meaning.

## Verification repeats the contract independently

Type inference validates source expressions, but SSA can also be built or
transformed directly. An optimization bug could replace an operand, invent a
wrong result type, or construct an invalid operation without going through
`infer_dot`.

The verifier therefore does not trust the frontend. For a `dot` operation it
checks:

```text
operand count == 2
lhs rank == 2
rhs rank == 2
lhs element == f32
rhs element == f32
lhs.shape[1] == rhs.shape[0]
result == block<lhs.rows x rhs.columns x f32>
```

This catches malformed hand-written SSA such as:

```text
%2 = dot %0, %1 : block<4x7 x f32>
```

when `%0` is `[4, 16]` and `%1` is `[16, 8]`. The operands describe a `[4, 8]`
result; declaring `[4, 7]` is an IR error even if the source frontend would
never emit it.

The duplicated-looking logic is intentional:

```text
source expression --type inference--> typed SSA
                                      |
                                      v
                              independent verifier
```

Inference constructs a type. Verification audits the resulting IR contract.

## `dot` is pure

`tl.dot` reads its operands and produces a value. It does not mutate memory or
change observable program state. Version 13 records that fact in the optimizer.

Common subexpression elimination may merge identical products:

```text
%2 = dot %0, %1
%3 = dot %0, %1
```

into one value, while dead-code elimination may remove a result that no store
or other live operation consumes.

This is another reason to add the operation through every compiler layer. If
the optimizer does not know whether a new opcode has side effects, it must
either keep useless work or risk deleting something observable.

## The AST frontend needs no special syntax

One benefit of the AST frontend from Part 9 is that an ordinary language call
already follows the same path as other `tl` functions:

```python
result = tl.dot(lhs, rhs)
```

The frontend resolves `tl.dot`, evaluates the call with symbolic arguments,
and receives a symbolic `Value`. There is no dedicated Python AST node for
matrix multiplication and no frontend-only representation of the operation.

This keeps the boundary clean:

```text
Python call syntax
      |
      v
public tl.dot function
      |
      v
Dot expression node
      |
      v
typed SSA dot
```

## Failing explicitly in CUDA

Version 13 intentionally has no CUDA lowering for `dot`. The backend says so:

```python
elif op.opcode == "dot":
    raise TypeError("CUDA lowering for tl.dot is not implemented")
```

An explicit failure is part of the implementation. Silently emitting a zero,
falling back to elementwise multiplication, or leaving the SSA value undefined
would turn a missing backend feature into a wrong kernel.

This also gives the next milestones a precise starting point. Version 14 can
make progress on data movement while preserving an honest diagnostic for the
still-missing computation.

## Testing the semantic boundary

The tests cover each layer separately.

Expression-tree tests check that:

- the public API creates a `Dot` node;
- both operands are preserved;
- the result remains symbolic.

Type-inference tests cover valid and degenerate shapes, non-block operands,
wrong ranks, mismatched `K` dimensions, and unsupported element types.

SSA tests check exact output:

```text
%0 = zeros {shape=(4, 16), dtype=f32} : block<4x16 x f32>
%1 = zeros {shape=(16, 8), dtype=f32} : block<16x8 x f32>
%2 = dot %0, %1 : block<4x8 x f32>
```

Verifier tests construct malformed operations directly so they do not merely
repeat frontend coverage. Optimizer tests establish purity through both CSE and
DCE. Finally, an AST-to-CUDA test checks that compilation reaches the explicit
unsupported-lowering diagnostic.

That last test is useful even though no kernel executes. It proves that the
entire frontend and middle end agree on `dot`; only the intended backend stage
is absent.

## What Version 13 does not do

Version 13 does not:

- allocate shared memory;
- recognize matrix load patterns;
- move `A` or `B` tiles cooperatively;
- emit a multiply-accumulate loop;
- assign output elements to registers;
- support mixed precision;
- use tensor cores.

Those omissions are not hidden behind the new API. `tl.dot` has real language,
type, SSA, verification, and optimization semantics, but CUDA compilation stops
with a clear error.

## What changed conceptually

Before Version 13, matrix multiplication was only an emergent pattern of small
operations:

```text
loads + elementwise arithmetic + reductions
```

After Version 13, it is an explicit compiler concept:

```text
rank-2 operands
      |
      v
typed and verified dot operation
      |
      v
backend-specific implementation point
```

That separation matters. The source language now says *what* product is
required without prematurely saying *how* CUDA must compute it.

All code for this milestone is available at
[https://github.com/pbelevich/mytriton/tree/ver13](https://github.com/pbelevich/mytriton/tree/ver13).

Next: [Part 14: Shared-Memory Tiles]({% post_url 2026-08-29-My_Triton_From_Scratch_Part_14_Shared_Memory_Tiles %}).
