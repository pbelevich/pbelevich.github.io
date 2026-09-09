---
layout: post
title:  "My Triton From Scratch Part 17: PyTorch Tensor Interoperability"
date:   2026-09-06 16:00:00 +0000
# categories:
---

In [Part 13: tl.dot Semantics]({% post_url 2026-08-23-My_Triton_From_Scratch_Part_13_Dot_Semantics %}),
matrix multiplication became a first-class compiler operation.

In [Part 14: Shared-Memory Tiles]({% post_url 2026-08-29-My_Triton_From_Scratch_Part_14_Shared_Memory_Tiles %}),
the CUDA backend learned to recognize and stage matrix tiles.

In [Part 15: CUDA-Core tl.dot]({% post_url 2026-08-30-My_Triton_From_Scratch_Part_15_CUDA_Core_Dot %}),
those tiles became a working CUDA-core matrix multiplication.

In [Part 16: Register Tiles]({% post_url 2026-09-05-My_Triton_From_Scratch_Part_16_Register_Tiles %}),
one CUDA thread learned to own several output elements.

All of those kernels accepted NumPy arrays for compile-only source inspection
and CuPy arrays for CUDA execution. That was enough to develop a compiler, but
it was an awkward boundary for the ecosystem in which Triton kernels are most
often used.

Version 17 lets the same kernels accept PyTorch tensors directly.

The interesting part is not recognizing another Python class. The interesting
part is preserving the low-level launch model:

```text
Torch owns the allocation and stream
CuPy still compiles and launches the generated CUDA
mytriton connects them without copying tensor data
```

That connection needs explicit rules for devices, frameworks, contiguity,
DLPack ownership, streams, and autograd.

## The kernel API should not change

The source kernel remains ordinary mytriton code:

```python
@triton.jit
def add_kernel(x, y, out, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x_values = tl.load(x + offsets, mask=mask)
    y_values = tl.load(y + offsets, mask=mask)

    tl.store(out + offsets, x_values + y_values, mask=mask)
```

Only the host-side values change:

```python
import torch

n = 1_000
block_size = 256

x = torch.ones(n, device="cuda", dtype=torch.float32)
y = torch.ones(n, device="cuda", dtype=torch.float32)
out = torch.empty_like(x)

add_kernel[((n + block_size - 1) // block_size,)](
    x,
    y,
    out,
    n,
    BLOCK_SIZE=block_size,
)
```

The AST frontend should not need a Torch-specific branch. Neither should type
inference, SSA verification, optimization, or CUDA source generation. A tensor
argument is still a pointer parameter:

```text
torch.Tensor<float32, cuda:0>
             |
             v
         ptr<f32>
```

Framework-specific work belongs at the runtime boundary.

## Describing arrays without importing every framework

Earlier code inferred an array from attributes such as `dtype` and `flags`.
That happened to fit NumPy and CuPy, but Torch exposes contiguity and device
metadata differently.

Version 17 adds one normalized description:

```python
@dataclass(frozen=True)
class RuntimeArrayInfo:
    framework: Literal["numpy", "cupy", "torch"]
    device: Literal["cpu", "cuda"]
    device_index: int | None
    dtype_name: str
    c_contiguous: bool

    @property
    def is_cuda(self) -> bool:
        return self.device == "cuda"
```

`array_arg_info(value)` translates each supported framework into this record.
NumPy obtains contiguity from `value.flags.c_contiguous`; Torch calls
`value.is_contiguous()`; CuPy exposes both its flags and CUDA device.

The compiler can now ask framework-independent questions:

```text
Is this an array?
Is it on CPU or CUDA?
Which CUDA device owns it?
What is its element dtype?
Is its memory C-contiguous?
```

The helper recognizes CuPy and Torch values from their type module and reads
the small metadata surface it needs. Importantly, importing `mytriton` does not
unconditionally import PyTorch. Torch remains an optional integration.

## Runtime types stay compiler types

Parameter construction uses the normalized metadata:

```python
array_info = array_arg_info(value)

if array_info is not None:
    if array_info.dtype_name != "float32":
        raise TypeError(...)
    if not array_info.c_contiguous:
        raise TypeError(...)
    return Param(name, PTR_F32)
```

In Version 17, the supported array element type is still only `float32`.
Low-precision pointer types are a later milestone.

The key property is that NumPy, CuPy, and Torch do not leak into typed SSA:

```text
NumPy float32 array -----\
CuPy float32 array -------+--> Param(name, ptr<f32>)
Torch float32 tensor ----/
```

The generated kernel signature is identical:

```cuda
extern "C" __global__
void add_kernel(float* x, float* y, float* out, int n) {
    ...
}
```

Changing the host container does not invalidate the language or backend
contract.

## CPU tensors remain compile-only

mytriton already used NumPy arrays as compile-only placeholders. Passing them
builds the expression tree, lowers and verifies SSA, and returns CUDA source,
but it does not try to launch a GPU kernel.

Torch CPU tensors follow the same rule:

```python
x = torch.ones(16, dtype=torch.float32)
y = torch.ones(16, dtype=torch.float32)
out = torch.empty(16, dtype=torch.float32)

_, _, cuda_src = add_kernel[(1,)](
    x,
    y,
    out,
    16,
    BLOCK_SIZE=16,
)
```

This path must work without importing CuPy. It is useful for source tests,
compiler debugging, and machines without CUDA.

The runtime decision is therefore based on array devices, not on whether CuPy
happens to be installed:

```text
only CPU arrays    -> compile, do not execute
any CUDA arrays    -> validate the complete launch, then execute
no array arguments -> compile, do not execute
```

## A launch needs one coherent memory world

Accepting several array frameworks creates combinations that should fail before
code generation or launch.

Version 17 rejects mixed CPU and CUDA arrays:

```text
Torch CPU input + Torch CUDA output -> error
```

A CUDA kernel cannot safely treat the CPU pointer as device-accessible storage.

It also rejects mixed CUDA frameworks in one launch:

```text
CuPy CUDA input + Torch CUDA output -> error
```

That restriction is conservative. CUDA pointers from both frameworks could in
principle coexist, but choosing one framework establishes the stream and
ownership protocol for the whole launch. Version 17 keeps that protocol
unambiguous.

Finally, every CUDA array must belong to the same device:

```text
cuda:0 input + cuda:1 output -> error
```

The validation produces three separate diagnostics because these are three
different mistakes:

```text
mixed CPU/CUDA devices
mixed CUDA frameworks
multiple CUDA device indices
```

## CuPy remains the internal launcher

Version 17 does not add a second CUDA compiler or kernel launcher. CuPy still
provides `RawKernel`, module loading, launch syntax, and the CUDA runtime bridge.

Before launch, Torch tensors are normalized to CuPy views:

```python
if info.framework == "torch":
    normalized.append(cp.from_dlpack(value.detach()))
```

The complete path is:

```text
Torch CUDA tensor
       |
       v
     detach
       |
       v
     DLPack
       |
       v
zero-copy CuPy view
       |
       v
CuPy RawKernel launch
```

`cp.from_dlpack` consumes the tensor's DLPack representation and creates an
array view over the same CUDA allocation. It does not copy all elements into a
new buffer. The generated kernel therefore reads and writes Torch-owned memory
directly.

This boundary also works for the experimental MLIR backend. Once Torch tensors
are normalized to CuPy views, the existing memref argument expansion can
construct allocated pointer, aligned pointer, offset, size, and stride values.

## Why `detach()` appears here

PyTorch may refuse a DLPack export from a tensor that requires gradients. A raw
CUDA launch is not a PyTorch autograd operation, so Version 17 detaches the
tensor before exporting it:

```python
cupy_view = cp.from_dlpack(tensor.detach())
```

`detach()` does not copy the storage. It creates a tensor view that shares the
same allocation but is not connected to the autograd graph.

That behavior is intentional:

```text
mytriton accepts a pointer to tensor storage
mytriton mutates output storage
mytriton does not create grad_fn
mytriton does not implement backward
```

This matches the role of a low-level Triton kernel. Autograd integration is a
separate PyTorch wrapper concern: a user could place forward and backward
kernels inside a custom `torch.autograd.Function`, but the compiler itself
should not invent differentiation semantics for an arbitrary kernel.

A test passes an input with `requires_grad=True`, verifies the numerical
result, and checks that the output did not unexpectedly acquire autograd
metadata.

## Stream correctness is part of correctness

Converting the pointer without respecting the current Torch stream would be a
subtle race.

Consider:

```python
stream = torch.cuda.Stream()

with torch.cuda.stream(stream):
    x = make_input()
    add_kernel[(grid,)](...)
    result = consume_output()
```

All three operations are ordered on `stream`. If CuPy silently launches the
kernel on an unrelated default stream, the kernel may read `x` before
`make_input()` completes or `consume_output()` may read `out` before the kernel
finishes.

Version 17 obtains the active Torch stream for the tensor's device:

```python
torch_stream = torch.cuda.current_stream(device=device_index)
```

It wraps the native handle as an external CuPy stream:

```python
with cp.cuda.Stream.from_external(torch_stream):
    # DLPack conversion and RawKernel launch
```

The launch is now inserted into the same CUDA command stream:

```text
Torch producer
      |
      v
mytriton/CuPy kernel
      |
      v
Torch consumer
```

No global `torch.cuda.synchronize()` is needed inside the runtime. Preserving
stream order is both safer and less disruptive than synchronizing the entire
device around every kernel.

## Device context and kernel caches

The runtime enters the CUDA device context selected by the array arguments:

```python
with cp.cuda.Device(device_index):
    ...
```

Compilation and launch therefore happen on the intended device even when it is
not device zero.

The CUDA kernel cache key also includes the device ID:

```text
(generated CUDA source, kernel name, device ID)
```

That avoids accidentally reusing a device-specific CuPy kernel object in a
different CUDA context.

The MLIR path similarly asks `cuda_chip(runtime_args)` for the architecture of
the tensor's actual device before compiling its cubin.

## Contiguity remains explicit

Version 17 accepts only C-contiguous arrays and tensors. A non-contiguous Torch
view may have the same logical shape but a very different stride:

```python
x = torch.zeros((8, 16), device="cuda").T
```

The current pointer arithmetic assumes dense row-major storage. Treating `x`
as contiguous would produce valid CUDA pointers to the wrong logical elements.

Rejecting it is preferable to silently copying it. An implicit
`tensor.contiguous()` would allocate storage and change the performance and
aliasing behavior of what looks like a low-level kernel call.

Future support for strided tensors should expose strides as real runtime
parameters and make indexing semantics explicit.

## Tests exercise the boundary without requiring Torch everywhere

Small fake array classes test metadata normalization and launch validation
without requiring a CUDA device. They cover:

- Torch CPU and CUDA recognition;
- C-contiguity;
- CPU-only compile behavior;
- mixed CPU/CUDA rejection;
- mixed-framework rejection;
- multiple-device rejection.

Real PyTorch tests then cover:

- compilation with CPU tensors without loading CuPy;
- CUDA execution through both CUDA and MLIR backends;
- tensors with `requires_grad=True`;
- observation of the current Torch stream inside the launcher;
- a complete producer-kernel-consumer chain on a non-default stream.

The stream tests matter as much as the numerical add test. A kernel that
usually returns the right numbers but races on non-default streams is not a
correct PyTorch integration.

## What Version 17 does not do

Version 17 does not:

- implement PyTorch autograd;
- register a custom Torch operator;
- accept non-contiguous tensors;
- allow mixed Torch and CuPy arguments in one launch;
- copy CPU tensors to CUDA automatically;
- support low-precision tensor dtypes yet;
- replace CuPy as the CUDA compiler and launcher.

The milestone is intentionally a runtime adapter, not a new frontend or a deep
framework integration.

## What changed conceptually

Before Version 17, CUDA execution was tied directly to one Python array type:

```text
CuPy array -> compiler parameter + CUDA launch
```

After Version 17, the compiler first normalizes host containers into runtime
facts:

```text
NumPy / CuPy / Torch
          |
          v
RuntimeArrayInfo
          |
          +--> compiler pointer type
          |
          +--> compile-only or execute
          |
          +--> device/framework validation
          |
          +--> DLPack + current-stream launch context
```

The source language still sees pointers. The compiler IR still sees
`ptr<f32>`. Only the edge that connects those pointers to real allocations now
understands PyTorch.

All code for this milestone is available at
[https://github.com/pbelevich/mytriton/tree/ver17](https://github.com/pbelevich/mytriton/tree/ver17).

Next: [Part 18: Shared-Memory Optimization]({% post_url 2026-09-07-My_Triton_From_Scratch_Part_18_Shared_Memory_Optimization %}).
