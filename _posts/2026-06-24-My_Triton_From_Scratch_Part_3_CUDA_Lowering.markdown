---
layout: post
title:  "My Triton From Scratch Part 3: CUDA Lowering"
date:   2026-06-24 12:00:00 +0000
# categories:
---

In [Part 1: Symbolic Tracing]({% post_url 2026-06-22-My_Triton_From_Scratch_Part_1_Symbolic_Tracing %}),
mytriton learned how to execute a Python function with symbolic values and
capture the result as an expression-tree IR.

In [Part 2: Typed SSA]({% post_url 2026-06-23-My_Triton_From_Scratch_Part_2_Typed_SSA %}),
it learned how to infer types for that tree and lower it into a small SSA-style
IR. The vector-add kernel stopped being one deeply nested `Store` and became a
linear sequence of typed operations with explicit inputs and results.

Version 3 adds the next compiler layer: it lowers that typed SSA to
[CUDA C++](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html).
The generated source can be inspected with ordinary NumPy inputs, or compiled
and launched through CuPy when the arguments live on a CUDA GPU.

The code for this milestone is here:
[https://github.com/pbelevich/mytriton/tree/ver3](https://github.com/pbelevich/mytriton/tree/ver3).

The GPU execution at the end is satisfying, but it is not the most interesting
part of this version. CuPy already knows how to call
[NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html) and launch a CUDA kernel.
The compiler-shaped work is deciding what each SSA operation means in CUDA,
especially when one SSA value represents an entire vector while one CUDA thread
holds only one scalar.

That lowering decision is the center of this part.

## The program we want to lower

The input is still the same small vector-add kernel:

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

Version 2 lowers it to:

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

This is a much better input for code generation than the original expression
tree. Operations are already in dependency order. Every result has a type.
Shared values such as `%3` and `%5` are explicit. The CUDA backend does not
need to rediscover any of that structure.

The launch now returns all three representations:

```python
expression_ops, ssa_ops, cuda_src = add_kernel[grid](
    x,
    y,
    out,
    n,
    BLOCK=256,
)
```

With NumPy arrays, this is a compile-only operation. `cuda_src` is still
produced, but no GPU library is imported and `out` is not modified. That makes
the backend easy to inspect and test on a machine without CUDA.

## The central lowering choice

The most important question is what this type means on a GPU:

```text
vector<256 x f32>
```

One possible backend could materialize a local C array with 256 elements. That
would be a literal interpretation of the type, but it would be a poor model of
the Triton program we are trying to imitate.

In Triton, a program instance conceptually operates on a block of values. The
compiler distributes those values across GPU threads. A Python variable can
look like one vector even though its lanes are held and computed by different
threads.

Version 3 uses the smallest possible version of that model:

- one mytriton program instance becomes one
  [CUDA block](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy);
- one vector lane becomes one CUDA thread in that block;
- a vector SSA value becomes one scalar local value per thread;
- `program_id(axis)` becomes `blockIdx.x`, `blockIdx.y`, or `blockIdx.z`;
- `arange(start, end)` becomes `start + threadIdx.x`.

For a vector width of 256, the kernel launches 256 threads per CUDA block.
Thread 0 represents lane 0, thread 1 represents lane 1, and so on.

This is intentionally much simpler than real Triton lowering. There is no warp
layout, no multidimensional tensor encoding, no vectorization inside a thread,
and no decision about how values should be distributed. The distribution is a
fixed rule: lane `i` belongs to CUDA thread `i`.

The rule is limited, but it gives every vector operation in the current IR a
clear scalar meaning.

## Vector types disappear from local declarations

Because one CUDA thread owns one lane, the CUDA type mapper strips the vector
shape before choosing a C++ type:

```python
def cuda_type(self, ty: Type) -> str:
    if isinstance(ty, VectorType):
        ty = ty.element

    if ty == I32:
        return "int"
    if ty == F32:
        return "float"
    if ty == BOOL:
        return "bool"
    if isinstance(ty, PointerType):
        return f"{self.cuda_type(ty.element)}*"

    raise TypeError(f"Cannot lower CUDA type: {ty}")
```

That gives the following mapping:

```text
i32                         -> int
f32                         -> float
bool                        -> bool
ptr<f32>                    -> float*
vector<256 x i32>           -> int per thread
vector<256 x f32>           -> float per thread
vector<256 x bool>          -> bool per thread
vector<256 x ptr<f32>>      -> one logical address per thread
```

The last case needs a little care. Numeric and Boolean values can become local
CUDA variables. A vector of pointers is more useful as a base pointer plus an
index expression, so the backend represents it symbolically instead of
materializing a pointer local immediately.

## A value table connects SSA to CUDA

The SSA operation list refers to values by numeric identity. The generated
CUDA source refers to locals such as `v3` and `v6`.

The code generator keeps a small table between those worlds:

```python
self.values: dict[int, str | CudaPtrRef] = {}
```

Most entries are strings naming CUDA locals:

```text
SSA %3 -> "v3"
SSA %5 -> "v5"
SSA %6 -> "v6"
```

Pointer values use a different representation:

```python
@dataclass(frozen=True)
class CudaPtrRef:
    base: str
    index: str
```

For example:

```text
SSA %4 -> CudaPtrRef(base="x", index="v3")
```

This says that `%4` refers to `x[v3]`. It is not yet a load, and it is not
numeric addition. It is an address waiting for a memory operation to consume
it.

The distinction from
[Part 1: Symbolic Tracing]({% post_url 2026-06-22-My_Triton_From_Scratch_Part_1_Symbolic_Tracing %})
has survived all the way through the compiler. `AddPtr` became the SSA opcode
`addptr`, and now `addptr` becomes a backend pointer reference rather than an
ordinary CUDA `+` expression.

## Looking up operands

SSA operands can be previous SSA values, kernel parameters, constants, or
`None` for omitted optional operands.

The backend handles those cases in one place:

```python
def operand(self, operand: SSAOperand) -> str | CudaPtrRef | None:
    if operand is None:
        return None
    if isinstance(operand, SSAValue):
        if operand.id not in self.values:
            raise RuntimeError(f"SSA value {operand} is not defined")
        return self.values[operand.id]
    if isinstance(operand, Param):
        return operand.name
    if isinstance(operand, Const):
        return self.literal(operand.value)

    raise TypeError(f"Unknown operand: {operand}")
```

The undefined-value check is small but useful. SSA lowering is supposed to
produce operations in dependency order. If code generation sees `%9` before an
entry for `%9` exists, either the IR is malformed or a lowering rule forgot to
record its result.

Two stricter helpers make the expected operand category explicit:

```python
def expression_operand(self, operand: SSAOperand) -> str:
    value = self.operand(operand)
    if not isinstance(value, str):
        raise TypeError(f"Expected CUDA expression, got {value}")
    return value


def pointer_operand(self, operand: SSAOperand) -> CudaPtrRef:
    value = self.operand(operand)
    if isinstance(value, str):
        return CudaPtrRef(value, "0")
    if not isinstance(value, CudaPtrRef):
        raise TypeError(f"Expected CUDA pointer, got {value}")
    return value
```

A raw pointer parameter such as `x` becomes `CudaPtrRef("x", "0")` when a
memory operation asks for it. A numeric expression cannot accidentally flow
into pointer lowering.

The type checker should already have rejected that program, but backend checks
are still valuable. A compiler is easier to debug when each layer defends its
own assumptions.

## Lowering program IDs

The first SSA operation is:

```text
%0 = program_id {axis=0} : i32
```

CUDA already has the corresponding block coordinates, so the lowering is
direct:

```python
if op.opcode == "program_id":
    axis = op.attrs["axis"]

    if axis not in (0, 1, 2):
        raise ValueError(f"Invalid program axis: {axis}")

    component = ("x", "y", "z")[axis]
    self.assign(result, f"blockIdx.{component}")
```

The `assign` helper creates a CUDA local using the SSA result type:

```python
def assign(self, result: SSAValue, expression: str) -> None:
    name = f"v{result.id}"
    cuda_ty = self.cuda_type(result.ty)
    self.lines.append(f"    {cuda_ty} {name} = {expression};")
    self.values[result.id] = name
```

For `%0`, it emits:

```cuda
int v0 = blockIdx.x;
```

Every thread in the block computes the same `v0`. That duplicates a tiny
amount of scalar work, but it preserves the right value and keeps the lowering
simple. A later optimizer or more sophisticated backend could treat values
that are uniform across a block differently from values that vary per thread.

Real GPU compilers care deeply about that distinction. Version 3 merely makes
it visible.

## Lowering a range to thread IDs

The vector-producing operation:

```text
%2 = arange {start=0, end=256} : vector<256 x i32>
```

is where the distributed-vector rule becomes concrete.

The whole SSA result is a vector, but each thread only needs its own lane:

```python
elif op.opcode == "arange":
    start = op.attrs["start"]
    expression = (
        "threadIdx.x"
        if start == 0
        else f"({start} + threadIdx.x)"
    )
    self.assign(result, expression)
```

For `arange(0, 256)`, the result is:

```cuda
int v2 = threadIdx.x;
```

Thread 0 gets the scalar value 0. Thread 137 gets 137. Taken together, the
threads represent the SSA vector `[0, 1, ..., 255]` without any thread storing
that complete vector.

A nonzero start works the same way:

```text
arange(4, 8)
```

becomes:

```cuda
int v0 = (4 + threadIdx.x);
```

The vector end is not printed into the expression. It determines the vector
width during type inference, and that width later determines how many threads
the block receives. The start affects the value held by each thread; the end
affects the launch shape.

This is a good example of one IR operation lowering partly into source and
partly into launch configuration.

## Arithmetic becomes scalar arithmetic per thread

The backend supports the small set of binary operations already present in the
SSA IR:

```python
BINARY_OPS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "cmp_lt": "<",
}
```

Lowering is mechanical:

```python
elif op.opcode in self.BINARY_OPS:
    lhs = self.expression_operand(op.operands[0])
    rhs = self.expression_operand(op.operands[1])
    symbol = self.BINARY_OPS[op.opcode]
    self.assign(result, f"({lhs} {symbol} {rhs})")
```

What makes it correct is not the dictionary. It is the vector-to-thread rule
established earlier.

Consider:

```text
%3 = add %1, %2 : vector<256 x i32>
```

`%1` is the scalar base offset for the block. `%2` is one range lane per
thread. CUDA emits:

```cuda
int v3 = (v1 + v2);
```

Each thread adds the same block base to its own lane ID, so all threads
together compute the vector of global offsets.

The same rule handles vector addition:

```text
%9 = add %6, %8 : vector<256 x f32>
```

which becomes:

```cuda
float v9 = (v6 + v8);
```

`v6`, `v8`, and `v9` are scalar locals in one thread. Across the block, they
represent the three vector SSA values.

## Pointer addition should not emit a local yet

The SSA contains:

```text
%4 = addptr x, %3 : vector<256 x ptr<f32>>
```

The naive lowering would emit something like:

```cuda
float* v4 = x + v3;
```

That would work for this example, but keeping the address as `base + index`
makes the later memory operations clearer and naturally handles nested pointer
addition.

The backend records a `CudaPtrRef` instead:

```python
elif op.opcode == "addptr":
    base = self.operand(op.operands[0])
    offset = self.expression_operand(op.operands[1])

    if isinstance(base, CudaPtrRef):
        if base.index != "0":
            offset = f"({base.index} + {offset})"
        base = base.base

    if not isinstance(base, str):
        raise TypeError(f"addptr expects pointer base, got {base}")

    self.values[result.id] = CudaPtrRef(base, offset)
```

After lowering `%4`, the table contains:

```text
%4 -> CudaPtrRef("x", "v3")
```

No source line is emitted yet.

If pointer additions are nested, their indices are folded:

```python
out + 1 + 2
```

eventually becomes:

```cuda
out[(1 + 2)]
```

This tiny symbolic address representation is effectively a backend-specific
mini-IR. It exists because a pointer value is more useful to this code
generator in decomposed form than as an eagerly named CUDA temporary.

That pattern appears often in compilers. Lowering does not always replace one
source operation with one target instruction. Sometimes it changes the way a
value is represented so that a later operation can emit better target code.

## Masked loads must not touch invalid memory

The first load is:

```text
%6 = load %4, %5, 0.0 : vector<256 x f32>
```

At this point, the backend has:

```text
%4 -> x[v3]
%5 -> v5
```

The desired per-thread behavior is:

```text
if this lane is valid:
    read x[v3]
else:
    use 0.0
```

CUDA's conditional operator expresses that directly:

```python
elif op.opcode == "load":
    ptr = self.pointer_operand(op.operands[0])
    mask_operand = op.operands[1]
    other_operand = op.operands[2]

    mask = (
        "true"
        if mask_operand is None
        else self.expression_operand(mask_operand)
    )

    if other_operand is None:
        other = (
            "0.0f"
            if self.scalar_type(result.ty) == F32
            else "0"
        )
    else:
        other = self.expression_operand(other_operand)

    self.assign(
        result,
        f"({mask} ? {ptr.base}[{ptr.index}] : {other})",
    )
```

The generated line is:

```cuda
float v6 = (v5 ? x[v3] : 0.0f);
```

The conditional operator evaluates only the selected branch. A thread with
`v5 == false` uses `0.0f` and does not evaluate `x[v3]`. That is the important
memory-safety property for the final partial block.

If the mask is absent, the backend substitutes `true`. If `other` is absent,
it uses a zero appropriate for the result element type. This lets masked and
unmasked loads share one lowering path.

The source is slightly more verbose for an unmasked load:

```cuda
float v0 = (true ? source[0] : 0.0f);
```

A CUDA compiler will simplify the constant condition. I prefer the uniform
lowering here because it keeps the semantics obvious and the implementation
small.

## Masked stores become control flow

A store has no SSA result, so it does not go through `assign`. It consumes an
address, a value, and an optional mask:

{% raw %}
```python
if op.opcode == "store":
    ptr = self.pointer_operand(op.operands[0])
    value = self.expression_operand(op.operands[1])
    mask_operand = op.operands[2]
    mask = (
        None
        if mask_operand is None
        else self.expression_operand(mask_operand)
    )

    if mask is None:
        self.lines.append(
            f"    {ptr.base}[{ptr.index}] = {value};"
        )
    else:
        self.lines.extend(
            [
                f"    if ({mask}) {{",
                f"        {ptr.base}[{ptr.index}] = {value};",
                "    }",
            ]
        )
    return
```
{% endraw %}

The final SSA operation:

```text
store %10, %9, %5
```

becomes:

```cuda
if (v5) {
    out[v3] = v9;
}
```

This introduces actual CUDA control flow even though the source language still
does not support symbolic Python `if` statements. The distinction matters.

The mask is data in the mytriton IR. The backend is free to realize that data
as a conditional expression, a branch, predicated instructions, or something
else with the same behavior. Version 3 chooses the most readable CUDA C++ form.

## Constants need target syntax too

Most constants are easy to print, but source generation still needs to respect
the target language.

Boolean values become `true` and `false`. Integer values use decimal syntax.
Finite Python floats get an `f` suffix so CUDA treats them as `float` rather
than `double`:

```python
return f"{value}f"
```

Special floating-point values need another representation:

```python
if math.isnan(value):
    return "__int_as_float(0x7fc00000)"

if math.isinf(value):
    infinity = "__int_as_float(0x7f800000)"
    return infinity if value > 0 else f"(-{infinity})"
```

This is a detail the expression and SSA layers did not need to care about.
They stored a Python float and its `f32` type. Only the CUDA backend needs to
decide how that value is spelled in generated source.

That is one of the useful boundaries provided by typed IR: frontend meaning
can remain independent of target syntax until the final lowering layer.

## Generating the kernel shell

Runtime parameters become the CUDA function signature:

```python
signature = ", ".join(
    f"{self.cuda_type(param.ty)} {param.name}"
    for param in params
)
```

For the add kernel:

```text
x   : ptr<f32> -> float* x
y   : ptr<f32> -> float* y
out : ptr<f32> -> float* out
n   : i32      -> int n
```

`BLOCK` is absent. It is a `constexpr` argument, so tracing already folded its
value into operations such as `mul %0, 256` and into the vector type produced
by `arange`. There is no runtime CUDA parameter left to pass.

The complete generation method is then almost boring:

{% raw %}
```python
def generate(self, kernel_name, ssa_ops, params) -> str:
    self.lines = []
    self.values = {}

    signature = ", ".join(
        f"{self.cuda_type(param.ty)} {param.name}"
        for param in params
    )

    for op in ssa_ops:
        self.emit(op)

    body = [
        'extern "C" __global__',
        f"void {kernel_name}({signature}) {{",
    ]
    body.extend(self.lines)
    body.append("}")

    return "\n".join(body)
```
{% endraw %}

The interesting behavior lives in the per-op lowering rules. The function
shell only resets state, walks the ordered SSA list, and wraps the emitted
lines in a CUDA kernel.

## The complete generated CUDA

Putting the lowering rules together produces:

```cuda
extern "C" __global__
void add_kernel(float* x, float* y, float* out, int n) {
    int v0 = blockIdx.x;
    int v1 = (v0 * 256);
    int v2 = threadIdx.x;
    int v3 = (v1 + v2);
    bool v5 = (v3 < n);
    float v6 = (v5 ? x[v3] : 0.0f);
    float v8 = (v5 ? y[v3] : 0.0f);
    float v9 = (v6 + v8);
    if (v5) {
        out[v3] = v9;
    }
}
```

The missing numbers are worth noticing. There is no local `v4`, `v7`, or
`v10`. Those SSA values are pointer references:

```text
%4  -> x[v3]
%7  -> y[v3]
%10 -> out[v3]
```

They were folded into the loads and store instead of becoming CUDA locals.

Everything else has a direct relationship to the typed SSA:

```text
%0 program_id -> blockIdx.x
%1 mul        -> v0 * 256
%2 arange     -> threadIdx.x
%3 add        -> v1 + v2
%5 cmp_lt     -> v3 < n
%6 load       -> masked x[v3]
%8 load       -> masked y[v3]
%9 add        -> v6 + v8
store         -> guarded out[v3] write
```

The generated code is not clever. That is a feature at this stage. I can put
the SSA and CUDA next to each other and account for every value.

## Choosing the CUDA block size from SSA

The source alone is not enough. The `arange` lowering assumes that every vector
lane has a corresponding `threadIdx.x`, so the launch must create exactly the
right number of threads.

Version 3 derives that number from the SSA result types:

```python
def _cuda_threads_per_block(ssa_ops: list[SSAOp]) -> int:
    vector_widths = {
        op.result.ty.size
        for op in ssa_ops
        if op.result is not None
        and isinstance(op.result.ty, VectorType)
    }

    if not vector_widths:
        return 1

    if len(vector_widths) != 1:
        rendered = ", ".join(
            str(width) for width in sorted(vector_widths)
        )
        raise ValueError(
            "CUDA lowering requires one vector width, "
            f"got: {rendered}"
        )

    threads_per_block = next(iter(vector_widths))
    if not 1 <= threads_per_block <= 1024:
        raise ValueError(...)

    return threads_per_block
```

A scalar-only kernel uses one thread per block. A kernel whose SSA values all
use `vector<256 x ...>` uses 256 threads.

A kernel that mixes 128-lane and 256-lane SSA values is rejected:

```text
CUDA lowering requires one vector width, got: 128, 256
```

That restriction follows directly from the simple lane mapping. One
`threadIdx.x` range cannot simultaneously represent two differently sized
distributed vectors without defining what the extra threads should do.

A more capable backend could attach layouts to values, predicate inactive
threads, or map multiple lanes to one thread. Version 3 does none of that, so
rejecting inconsistent widths is safer than quietly generating the wrong
program.

This is another compiler lesson I keep encountering: a lowering rule and its
legality checks are two halves of the same feature.

## Why generate CUDA C++ instead of PTX?

The project is trying to expose compiler layers, so stopping at readable CUDA
C++ is useful.

It lets the backend express the semantics we care about with familiar target
constructs:

- `blockIdx` for program IDs;
- `threadIdx` for vector lanes;
- scalar C++ arithmetic for per-lane operations;
- array indexing for pointer references;
- conditional expressions and `if` statements for masks.

Generating PTX directly would add another large set of details: virtual
register classes, address spaces, instruction suffixes, parameter loading,
predicates, and target-specific calling conventions. Those details are
interesting, but they are not necessary to make the SSA-to-thread mapping
concrete.

NVRTC can lower the readable CUDA source through the remaining NVIDIA compiler
stack. mytriton remains responsible for the language semantics it introduced,
while existing tools handle machine-level code generation.

This is not avoiding a compiler backend. It is choosing the backend boundary
that makes the current lesson easiest to inspect.

## Keeping execution intentionally thin

Once the source and launch dimensions exist, execution is mostly plumbing.
CuPy provides
[`RawKernel`](https://docs.cupy.dev/en/stable/reference/generated/cupy.RawKernel.html),
which compiles CUDA source with NVRTC and exposes a Python launch interface.

The essential bridge is:

```python
kernel = cp.RawKernel(
    cuda_src,
    kernel_name,
    options=("--std=c++14",),
)

kernel(
    launch_grid,
    (threads_per_block,),
    runtime_args,
)
```

There are a few practical checks around those lines:

- CuPy is imported lazily, so compile-only NumPy use does not require it;
- all runtime arrays must be either NumPy or CuPy, never a mixture;
- Python integer and float arguments are converted to `int32` and `float32`;
- the requested block size is checked against the current device limit;
- CUDA execution only happens when the runtime array arguments are CuPy arrays.

That code matters, but it does not define the compiler model. It transports
already-lowered source, dimensions, and arguments to the CUDA runtime. I want
that boundary to stay small enough that the interesting decisions remain in
`SSACUDACodegen`.

## Compilation now needs caching

Tracing, SSA lowering, CUDA source generation, and NVRTC compilation should not
happen again for every launch with the same specialization.

Version 3 therefore has two caches.

The first cache stores a compilation artifact:

```python
@dataclass(frozen=True)
class _CompilationArtifact:
    ops: tuple[Any, ...]
    ssa_ops: tuple[SSAOp, ...]
    cuda_src: str
    threads_per_block: int
```

Its key contains runtime parameter types and `constexpr` values. Array contents
and ordinary runtime scalar values do not affect compilation. `BLOCK=128` and
`BLOCK=256` produce different artifacts because they change both constants and
vector widths in the IR.

The launch grid is not part of the compilation key. The same generated kernel
can run with a different number of CUDA blocks.

The second cache stores CuPy `RawKernel` objects by generated source and kernel
name. That avoids asking NVRTC to compile identical CUDA repeatedly.

The current cache model is deliberately small. Python globals and closure
values used during tracing are not tracked. If they change, the caller must use:

```python
kernel.clear_cache()
```

A production JIT needs a much more careful definition of specialization state.
For this version, the cache is enough to separate compile-time work from launch
time work and make repeated execution behave like a JIT rather than a source
generator invoked on every call.

## Testing the backend without a GPU

Most CUDA lowering tests do not need CUDA.

The main test calls the kernel with NumPy arrays and compares the complete
generated source with the expected string. That checks the exact lowering of
program IDs, ranges, arithmetic, masks, pointer references, loads, and the
store.

Smaller tests cover the edges separately:

- a scalar unmasked load and store;
- nested pointer arithmetic;
- an `arange` with a nonzero start;
- `NaN` and positive and negative infinity literals;
- rejection of inconsistent vector widths;
- rejection of invalid launch dimensions;
- cache hits and misses for runtime types and `constexpr` values;
- validation of runtime arrays even when compilation hits the cache.

This is one reason returning `cuda_src` is useful even after execution exists.
Generated code remains a first-class artifact that can be inspected and tested
on an ordinary CPU CI runner.

There is also one end-to-end CUDA test. It creates CuPy arrays, launches the add
kernel, synchronizes the device, and compares the result with `x + y`. The test
skips when CUDA is unavailable, but a self-hosted GPU runner can set
`MYTRITON_REQUIRE_CUDA=1` to turn the skip into a failure.

The GPU test answers an important final question: does the source accepted by
the snapshot test also compile, launch, and produce the correct values?

It is one test because execution is the end of this milestone, not its largest
implementation surface.

## The complete path now runs

With CuPy arrays, the vector-add path is finally end to end:

```text
Python kernel
    -> symbolic expression-tree IR
    -> type inference
    -> typed SSA IR
    -> CUDA C++ lowering
    -> NVRTC compilation through CuPy
    -> CUDA kernel launch
    -> values written to out
```

The launch still looks like the API established in
[Part 1: Symbolic Tracing]({% post_url 2026-06-22-My_Triton_From_Scratch_Part_1_Symbolic_Tracing %}):

```python
add_kernel[
    lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
](
    x,
    y,
    out,
    n,
    BLOCK=256,
)
```

The difference is that `x`, `y`, and `out` can now be CuPy arrays. The grid
determines the CUDA block grid, the SSA vector width determines the CUDA thread
block, and the generated kernel performs the work.

That is a small execution model, but every part of it now has a visible origin
in the compiler:

- the grid comes from the launch API;
- the thread count comes from vector types;
- the kernel arguments come from runtime parameters;
- the source body comes from ordered SSA operations;
- the memory guards come from load and store masks.

Nothing important is inferred inside the CuPy call. By the time execution
begins, the compiler decisions have already been made.

## What's next

The next step is not to make the execution wrapper bigger. It is to make the
language slightly less like a vector-add demo.

Vector addition exercises a useful path through the compiler, but it is still a
very narrow one: a couple of binary operators, one comparison, two loads, and
one store. The next interesting milestone is to add just enough math to write
the first nonlinear kernels.

The operations I want next are small:

- unary negation;
- [`tl.exp`](https://triton-lang.org/main/python-api/generated/triton.language.exp.html);
- [`tl.minimum`](https://triton-lang.org/main/python-api/generated/triton.language.minimum.html);
- [`tl.maximum`](https://triton-lang.org/main/python-api/generated/triton.language.maximum.html).

They are small enough to fit in the current compiler, but each one touches the
whole stack. The frontend needs new expression nodes. Type inference has to
decide what the operation means for scalars and vectors. SSA lowering needs new
opcodes. CUDA codegen needs to print correct target expressions. The tests need
to check both the symbolic lowering and the generated source.

That is exactly the kind of change that should keep the layers honest.

`maximum` gives a natural ReLU kernel:

```python
value = tl.load(x + offsets, mask=mask, other=0.0)
tl.store(out + offsets, tl.maximum(value, 0.0), mask=mask)
```

Negation and `tl.exp` give a small sigmoid kernel:

```python
value = tl.load(x + offsets, mask=mask, other=0.0)
result = 1.0 / (1.0 + tl.exp(-value))
tl.store(out + offsets, result, mask=mask)
```

Those two examples are more useful than they look. ReLU forces the compiler to
lower an operation that is not spelled as a Python operator. Sigmoid forces
several expression kinds to compose: unary negation, a library-style math op,
addition, division, and a masked store.

Minimum and maximum also have real
[floating-point semantics](https://en.wikipedia.org/wiki/IEEE_754) hiding
inside their friendly names. I do not want the backend to accidentally choose
whatever C++ helper happens to compile. The lowering should say what happens
with [`NaN`](https://en.wikipedia.org/wiki/NaN), and it should make tie behavior
visible in tests.

After that, the larger compiler questions will still be waiting. A verifier,
function-level IR, optimization passes, richer layouts, reductions, shared
memory, and scheduling decisions are all still outside this tiny implementation.
But before building those bigger structures, I want a few more operations that
make the existing pipeline feel less like it was custom-built for one kernel.

All code for this milestone is available at
[https://github.com/pbelevich/mytriton/tree/ver3](https://github.com/pbelevich/mytriton/tree/ver3).

Next: [Part 4: Elementwise Ops]({% post_url 2026-06-25-My_Triton_From_Scratch_Part_4_Elementwise_Ops %}).
