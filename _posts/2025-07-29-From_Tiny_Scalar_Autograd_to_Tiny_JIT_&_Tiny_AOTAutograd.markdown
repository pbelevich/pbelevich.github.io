---
layout: post
title:  "From Tiny Scalar Autograd to Tiny JIT & Tiny AOTAutograd"
date:   2025-07-29 11:12:14 -0400
# categories: jit aot autograd
---
# From Tiny Scalar Autograd to Tiny JIT & Tiny AOTAutograd

Building a miniature autograd-and-compiler stack from scratch is a great way to demystify the layers that power modern machine-learning frameworks. In this post we'll start with a palm-sized eager-mode autograd engine—just enough operator overloading to record a computation graph and walk it backward for gradients—then progressively bolt on a tiny tracing JIT, and finally an ahead-of-time (AOT) pass that emits separate, fully compiled forward and backward functions. The goal isn't to replicate the formidable engineering that underpins torch.compile; rather, it's to give you a stripped-down, readable playground that echoes the big ideas while steering clear of the production-grade complexity hidden inside real PyTorch.

## 0. Start

We start our journey with the humblest of Python scripts:


```python
a = 1
b = 2
c = 3
d = -a + b * (c - a / b)
print(f"{d=}")
```

    d=4.0


Here everything is a plain scalar, so each arithmetic operator executes immediately, yielding the final value 4. There's no graph, no tracing, no gradients—just Python's interpreter evaluating an expression tree from right to left according to precedence rules. This bare-bones baseline makes an ideal launchpad: by swapping these raw numbers for objects that remember how they were produced, we'll be able to teach the program to differentiate itself, then capture the trace for JIT compilation, and eventually generate standalone forward- and backward-pass code.

## 1. Eager forward

In this first transformation, we swap out raw Python scalars for instances of a tiny Variable class that overloads the arithmetic operators. The numerical recipe stays the same: d = -a + b * (c - a / b)-but every +, -, *, /, and unary - now calls a method that constructs and returns a brand-new Variable. Under the hood we're no longer just crunching floats; we're wrapping each intermediate result in an object that can store extra bookkeeping (think: gradient slots, symbolic expression trees, or pointers to a computational graph). In other words, we've laid the groundwork for automatic differentiation without touching the high-level math: the business logic remains crystal-clear, while the machinery for tracking operations has been slipped in invisibly through Python's operator-overloading magic.


```python
class Variable:

    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return str(self.data)

    def __str__(self):
        return str(self.data)

    def __add__(self, other: "Variable") -> "Variable":
        return Variable(self.data + other.data)

    def __sub__(self, other: "Variable") -> "Variable":
        return Variable(self.data - other.data)

    def __mul__(self, other: "Variable") -> "Variable":
        return Variable(self.data * other.data)

    def __truediv__(self, other: "Variable") -> "Variable":
        return Variable(self.data / other.data)

    def __neg__(self) -> "Variable":
        return Variable(-self.data)

a = Variable(1)
b = Variable(2)
c = Variable(3)
d = -a + b * (c - a / b)
print(f"{d=}")

```

    d=4.0


## 2. Eager backward

To turn our inert wrapper into a true eager-mode autograd engine, we graft three new limbs onto `Variable`: **`args`** to remember the operands that produced a value, **`grad_fn`** to hold a tiny function that maps an incoming gradient to gradients for those operands, and **`grad`** to accumulate the result of back-propagation. Every overloaded operator now fabricates a fresh `Variable` that carries—alongside its numerical data-a closure capturing the local derivative rule (for `*`, the rule is ∂(xy)=y dx + x dy, encoded as a lambda). The new `backward` method seeds the output with a gradient of 1, walks recursively through the saved operands, and, at each step, hashes the chain rule by calling `grad_fn`, fanning the upstream gradient into child gradients before descending further. In one swoop we've transformed pure arithmetic ornamentation into a fully functional reverse-mode autodiff system that can compute `d`, call `d.backward()`, and deposit the correct sensitivities in `a.grad`, `b.grad`, and `c.grad`.



```python
class Variable:

    def __init__(self, data, args=None, grad_fn=None, grad=None):
        self.data = data
        self.args = args
        self.grad_fn = grad_fn
        self.grad = grad

    def __repr__(self):
        return str(self.data)

    def __str__(self):
        return str(self.data)

    def __add__(self, other: "Variable") -> "Variable":
        return Variable(self.data + other.data, args=[self, other], grad_fn=lambda grad: [grad, grad])

    def __sub__(self, other: "Variable") -> "Variable":
        return Variable(self.data - other.data, args=[self, other], grad_fn=lambda grad: [grad, -grad])

    def __mul__(self, other: "Variable") -> "Variable":
        return Variable(self.data * other.data, args=[self, other], grad_fn=lambda grad: [grad * other, self * grad])

    def __truediv__(self, other: "Variable") -> "Variable":
        return Variable(self.data / other.data, args=[self, other], grad_fn=lambda grad: [grad / other, -grad * self / other / other])

    def __neg__(self) -> "Variable":
        return Variable(-self.data, args=[self], grad_fn=lambda grad: [-grad])

    def backward(self, grad=None):
        if grad is None:
            grad = Variable(1)
        self.grad = grad if self.grad == None else self.grad + grad
        if self.grad_fn is not None and self.args is not None:
            for p, g in zip(self.args, self.grad_fn(grad)):
                p.backward(g)

a = Variable(1)
b = Variable(2)
c = Variable(3)
d = -a + b * (c - a / b)
print(f"{d=}")

d.backward()
print(f"{a.grad=}")
print(f"{b.grad=}")
print(f"{c.grad=}")

```

    d=4.0
    a.grad=-2.0
    b.grad=3.0
    c.grad=2


## 3. Symbolic forward

Next, we trade immediacy for introspection by replacing eager arithmetic with **symbolic execution**. Every `Variable` is now just a node in an expression tree whose `op` field names a tiny operator class (`Add`, `Mul`, `Neg`, …) and whose `args` field holds its children. Each operator class supplies two static utilities: `stringify`, which turns the subtree into human‑readable math, and `evaluate`, which—given a dictionary of concrete inputs—recursively walks the tree to compute a numeric result. Because no calculation happens when we build the tree, constructing

```python
a = Variable(Input, ['a'])
b = Variable(Input, ['b'])
c = Variable(Input, ['c'])
d = -a + b * (c - a / b)
```

merely pieces together a symbolic object that prints as “`(-a + (b * (c - (a / b))))`”. Only when we call `d.evaluate({'a': 1, 'b': 2, 'c': 3})` do the recursive `evaluate` methods descend the tree, substitute the dictionary values, and return `4`. This decoupling of *definition* from *execution* gives us a manipulable IR—perfect fodder for tracing, code generation, and ahead‑of‑time differentiation in the steps to come.



```python
class Input:
    def stringify(args):
        return str(args[0])

    def evaluate(args, inputs):
        return inputs[args[0]]

class Add:
    def stringify(args):
        return f"({args[0]} + {args[1]})"

    def evaluate(args, inputs):
        return args[0].evaluate(inputs) + args[1].evaluate(inputs)

class Sub:
    def stringify(args):
        return f"({args[0]} - {args[1]})"

    def evaluate(args, inputs):
        return args[0].evaluate(inputs) - args[1].evaluate(inputs)

class Mul:
    def stringify(args):
        return f"({args[0]} * {args[1]})"

    def evaluate(args, inputs):
        return args[0].evaluate(inputs) * args[1].evaluate(inputs)

class Div:
    def stringify(args):
        return f"({args[0]} / {args[1]})"

    def evaluate(args, inputs):
        return args[0].evaluate(inputs) / args[1].evaluate(inputs)

class Neg:
    def stringify(args):
        return f"(-{args[0]})"

    def evaluate(args, inputs):
        return -args[0].evaluate(inputs)

class Variable:

    def __init__(self, op, args):
        self.op = op
        self.args = args

    def __repr__(self):
        return self.op.stringify(self.args)

    def __str__(self):
        return self.op.stringify(self.args)

    def __add__(self, other: "Variable") -> "Variable":
        return Variable(Add, [self, other])

    def __sub__(self, other: "Variable") -> "Variable":
        return Variable(Sub, [self, other])

    def __mul__(self, other: "Variable") -> "Variable":
        return Variable(Mul, [self, other])

    def __truediv__(self, other: "Variable") -> "Variable":
        return Variable(Div, [self, other])

    def __neg__(self) -> "Variable":
        return Variable(Neg, [self])

    def evaluate(self, inputs):
        return self.op.evaluate(self.args, inputs)

a = Variable(Input, ['a'])
b = Variable(Input, ['b'])
c = Variable(Input, ['c'])
d = -a + b * (c - a / b)
print(f"{d=}")
print(f"{d.evaluate({'a': 1, 'b': 2, 'c': 3})=}")
```

    d=((-a) + (b * (c - (a / b))))
    d.evaluate({'a': 1, 'b': 2, 'c': 3})=4.0


## 4. Symbolic backward

To upgrade our symbolic IR into a **symbolic‑autodiff** engine we graft the same ideas we used in eager mode—saved operands and local gradient rules—onto the operator‑tree representation instead of onto concrete numbers. Each `Variable` now carries a `grad_fn` closure that spits out *symbolic* gradient nodes (still `Variable` objects) when given an upstream gradient, plus a `grad` slot to store the accumulated result. Arithmetic overloads still return a fresh forward node (`Add`, `Mul`, …), but they also capture the corresponding chain‑rule recipe—e.g. `λ grad → [grad, grad]` for addition or `λ grad → [grad * other, self * grad]` for multiplication. Calling `d.backward(Variable(Input, ['d_grad']))` seeds the output with a placeholder leaf named `d_grad`, then walks the expression tree in reverse, weaving together new symbolic sub‑trees for every partial derivative until each input leaf (`a`, `b`, `c`) owns its own gradient expression. Because those gradients are themselves `Variable`s, we can delay evaluation: `a.grad.evaluate({'a':1,'b':2,'c':3,'d_grad':1})` eventually produces the numeric sensitivity, but we could just as easily print or transform the expression before plugging in values. In short, we’ve fused forward‑pass algebra and backward‑pass algebra into a single manipulable graph—laying the groundwork for emitting *compiled* forward and backward code in the next stage.



```python
class Input:
    def stringify(args):
        return str(args[0])

    def evaluate(args, inputs):
        return inputs[args[0]]

class Add:
    def stringify(args):
        return f"({args[0]} + {args[1]})"

    def evaluate(args, inputs):
        return args[0].evaluate(inputs) + args[1].evaluate(inputs)

class Sub:
    def stringify(args):
        return f"({args[0]} - {args[1]})"

    def evaluate(args, inputs):
        return args[0].evaluate(inputs) - args[1].evaluate(inputs)

class Mul:
    def stringify(args):
        return f"({args[0]} * {args[1]})"

    def evaluate(args, inputs):
        return args[0].evaluate(inputs) * args[1].evaluate(inputs)

class Div:
    def stringify(args):
        return f"({args[0]} / {args[1]})"

    def evaluate(args, inputs):
        return args[0].evaluate(inputs) / args[1].evaluate(inputs)

class Neg:
    def stringify(args):
        return f"(-{args[0]})"

    def evaluate(args, inputs):
        return -args[0].evaluate(inputs)

class Variable:

    def __init__(self, op, args, grad_fn=None, grad=None):
        self.op = op
        self.args = args
        self.grad_fn = grad_fn
        self.grad = grad

    def __repr__(self):
        return self.op.stringify(self.args)

    def __str__(self):
        return self.op.stringify(self.args)

    def __add__(self, other: "Variable") -> "Variable":
        return Variable(Add, [self, other], grad_fn=lambda grad: [grad, grad])

    def __sub__(self, other: "Variable") -> "Variable":
        return Variable(Sub, [self, other], grad_fn=lambda grad: [grad, -grad])

    def __mul__(self, other: "Variable") -> "Variable":
        return Variable(Mul, [self, other], grad_fn=lambda grad: [grad * other, self * grad])

    def __truediv__(self, other: "Variable") -> "Variable":
        return Variable(Div, [self, other], grad_fn=lambda grad: [grad / other, -grad * self / other / other])

    def __neg__(self) -> "Variable":
        return Variable(Neg, [self], grad_fn=lambda grad: [-grad])

    def evaluate(self, inputs):
        return self.op.evaluate(self.args, inputs)

    def backward(self, grad):
        self.grad = grad if self.grad == None else self.grad + grad
        if self.op != Input:
            for p, g in zip(self.args, self.grad_fn(grad)):
                p.backward(g)


a = Variable(Input, ['a'])
b = Variable(Input, ['b'])
c = Variable(Input, ['c'])
d = -a + b * (c - a / b)
d.backward(Variable(Input, ['d_grad']))

d_evaluated = d.evaluate({'a': 1, 'b': 2, 'c': 3})
a_grad = a.grad.evaluate({'a': 1, 'b': 2, 'c': 3, 'd_grad': 1})
b_grad = b.grad.evaluate({'a': 1, 'b': 2, 'c': 3, 'd_grad': 1})
c_grad = c.grad.evaluate({'a': 1, 'b': 2, 'c': 3, 'd_grad': 1})

print(f"{d=} = {d_evaluated}")
print(f"{a.grad=} = {a_grad}")
print(f"{b.grad=} = {b_grad}")
print(f"{c.grad=} = {c_grad}")
```

    d=((-a) + (b * (c - (a / b)))) = 4.0
    a.grad=((-d_grad) + ((-(b * d_grad)) / b)) = -2.0
    b.grad=((d_grad * (c - (a / b))) + ((((-(-(b * d_grad))) * a) / b) / b)) = 3.0
    c.grad=(b * d_grad) = 2


## 5. LLVM IR

Finally, we strap a **code‑generation engine** onto our symbolic‑autodiff trees so they can be *lowered* straight to machine code instead of being interpreted at run time. Each operator class now exposes an `ir` method that emits the LLVM instructions—via llvmlite’s `IRBuilder`—needed to compute its value (`fadd`, `fmul`, `fneg`, …). A new `Variable.ir` simply delegates to that method, while `Variable.ir_module` wraps the whole expression in a top‑level LLVM function complete with named arguments and a `ret`. Because we run `backward` *before* calling `ir_module`, every input’s gradient is itself an expression tree, so we can compile **both** the forward scalar `d` and the three derivative functions `a_grad`, `b_grad`, and `c_grad` into standalone LLVM modules. In effect we’ve moved from a lazy symbolic interpreter to a tiny ahead‑of‑time compiler that spits out optimized IR for the forward pass and all Jacobian‑vector products, mirroring at toy scale what big frameworks do when they hand traces off to back‑ends like XLA, NVRTC, or the new PyTorch `torch.compile` pipeline.



```python
from llvmlite import ir

double = ir.DoubleType()

class Input:
    def stringify(args):
        return str(args[0])

    def evaluate(args, inputs):
        return inputs[args[0]]

    def ir(args, builder, func, inputs_keys):
        return func.args[inputs_keys.index(args[0])]

class Add:
    def stringify(args):
        return f"({args[0]} + {args[1]})"

    def evaluate(args, inputs):
        return args[0].evaluate(inputs) + args[1].evaluate(inputs)

    def ir(args, builder, func, inputs_keys):
        lhs = args[0].ir(builder, func, inputs_keys)
        rhs = args[1].ir(builder, func, inputs_keys)
        return builder.fadd(lhs, rhs, name=Add.stringify(args))

class Sub:
    def stringify(args):
        return f"({args[0]} - {args[1]})"

    def evaluate(args, inputs):
        return args[0].evaluate(inputs) - args[1].evaluate(inputs)

    def ir(args, builder, func, inputs_keys):
        lhs = args[0].ir(builder, func, inputs_keys)
        rhs = args[1].ir(builder, func, inputs_keys)
        return builder.fsub(lhs, rhs, name=Sub.stringify(args))

class Mul:
    def stringify(args):
        return f"({args[0]} * {args[1]})"

    def evaluate(args, inputs):
        return args[0].evaluate(inputs) * args[1].evaluate(inputs)

    def ir(args, builder, func, inputs_keys):
        lhs = args[0].ir(builder, func, inputs_keys)
        rhs = args[1].ir(builder, func, inputs_keys)
        return builder.fmul(lhs, rhs, name=Mul.stringify(args))

class Div:
    def stringify(args):
        return f"({args[0]} / {args[1]})"

    def evaluate(args, inputs):
        return args[0].evaluate(inputs) / args[1].evaluate(inputs)

    def ir(args, builder, func, inputs_keys):
        lhs = args[0].ir(builder, func, inputs_keys)
        rhs = args[1].ir(builder, func, inputs_keys)
        return builder.fdiv(lhs, rhs, name=Div.stringify(args))

class Neg:
    def stringify(args):
        return f"(-{args[0]})"

    def evaluate(args, inputs):
        return -args[0].evaluate(inputs)

    def ir(args, builder, func, inputs_keys):
        return builder.fneg(args[0].ir(builder, func, inputs_keys), name=Neg.stringify(args))

class Variable:

    def __init__(self, op, args, grad_fn=None, grad=None):
        self.op = op
        self.args = args
        self.grad_fn = grad_fn
        self.grad = grad

    def __repr__(self):
        return self.op.stringify(self.args)

    def __str__(self):
        return self.op.stringify(self.args)

    def __add__(self, other: "Variable") -> "Variable":
        return Variable(Add, [self, other], grad_fn=lambda grad: [grad, grad])

    def __sub__(self, other: "Variable") -> "Variable":
        return Variable(Sub, [self, other], grad_fn=lambda grad: [grad, -grad])

    def __mul__(self, other: "Variable") -> "Variable":
        return Variable(Mul, [self, other], grad_fn=lambda grad: [grad * other, self * grad])

    def __truediv__(self, other: "Variable") -> "Variable":
        return Variable(Div, [self, other], grad_fn=lambda grad: [grad / other, -grad * self / other / other])

    def __neg__(self) -> "Variable":
        return Variable(Neg, [self], grad_fn=lambda grad: [-grad])

    def evaluate(self, inputs):
        return self.op.evaluate(self.args, inputs)

    def backward(self, grad):
        self.grad = grad if self.grad == None else self.grad + grad
        if self.op != Input:
            for p, g in zip(self.args, self.grad_fn(grad)):
                p.backward(g)

    def ir(self, builder, func, inputs_keys):
        return self.op.ir(self.args, builder, func, inputs_keys)

    def ir_module(self, func_name, return_type, func_signature):
        fnty = ir.FunctionType(return_type, func_signature.values())
        module = ir.Module(name=func_name)
        func = ir.Function(module, fnty, name=func_name)
        block = func.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)
        builder.ret(self.ir(builder, func, list(func_signature.keys())))
        return module

a = Variable(Input, ['a'])
b = Variable(Input, ['b'])
c = Variable(Input, ['c'])
d = -a + b * (c - a / b)
d.backward(Variable(Input, ['d_grad']))

print(d.ir_module('d', double, {'a': double, 'b': double, 'c': double}))
print(a.grad.ir_module('a_grad', double, {'a': double, 'b': double, 'c': double, 'd_grad': double}))
print(b.grad.ir_module('b_grad', double, {'a': double, 'b': double, 'c': double, 'd_grad': double}))
print(c.grad.ir_module('c_grad', double, {'a': double, 'b': double, 'c': double, 'd_grad': double}))
```

    ; ModuleID = "d"
    target triple = "unknown-unknown-unknown"
    target datalayout = ""
    
    define double @"d"(double %".1", double %".2", double %".3")
    {
    entry:
      %"(-a)" = fneg double %".1"
      %"(a / b)" = fdiv double %".1", %".2"
      %"(c - (a / b))" = fsub double %".3", %"(a / b)"
      %"(b * (c - (a / b)))" = fmul double %".2", %"(c - (a / b))"
      %"((-a) + (b * (c - (a / b))))" = fadd double %"(-a)", %"(b * (c - (a / b)))"
      ret double %"((-a) + (b * (c - (a / b))))"
    }
    
    ; ModuleID = "a_grad"
    target triple = "unknown-unknown-unknown"
    target datalayout = ""
    
    define double @"a_grad"(double %".1", double %".2", double %".3", double %".4")
    {
    entry:
      %"(-d_grad)" = fneg double %".4"
      %"(b * d_grad)" = fmul double %".2", %".4"
      %"(-(b * d_grad))" = fneg double %"(b * d_grad)"
      %"((-(b * d_grad)) / b)" = fdiv double %"(-(b * d_grad))", %".2"
      %"((-d_grad) + ((-(b * d_grad)) / b))" = fadd double %"(-d_grad)", %"((-(b * d_grad)) / b)"
      ret double %"((-d_grad) + ((-(b * d_grad)) / b))"
    }
    
    ; ModuleID = "b_grad"
    target triple = "unknown-unknown-unknown"
    target datalayout = ""
    
    define double @"b_grad"(double %".1", double %".2", double %".3", double %".4")
    {
    entry:
      %"(a / b)" = fdiv double %".1", %".2"
      %"(c - (a / b))" = fsub double %".3", %"(a / b)"
      %"(d_grad * (c - (a / b)))" = fmul double %".4", %"(c - (a / b))"
      %"(b * d_grad)" = fmul double %".2", %".4"
      %"(-(b * d_grad))" = fneg double %"(b * d_grad)"
      %"(-(-(b * d_grad)))" = fneg double %"(-(b * d_grad))"
      %"((-(-(b * d_grad))) * a)" = fmul double %"(-(-(b * d_grad)))", %".1"
      %"(((-(-(b * d_grad))) * a) / b)" = fdiv double %"((-(-(b * d_grad))) * a)", %".2"
      %"((((-(-(b * d_grad))) * a) / b) / b)" = fdiv double %"(((-(-(b * d_grad))) * a) / b)", %".2"
      %"((d_grad * (c - (a / b))) + ((((-(-(b * d_grad))) * a) / b) / b))" = fadd double %"(d_grad * (c - (a / b)))", %"((((-(-(b * d_grad))) * a) / b) / b)"
      ret double %"((d_grad * (c - (a / b))) + ((((-(-(b * d_grad))) * a) / b) / b))"
    }
    
    ; ModuleID = "c_grad"
    target triple = "unknown-unknown-unknown"
    target datalayout = ""
    
    define double @"c_grad"(double %".1", double %".2", double %".3", double %".4")
    {
    entry:
      %"(b * d_grad)" = fmul double %".2", %".4"
      ret double %"(b * d_grad)"
    }
    


# 6. LLVM compiled forward and backward

In the last step we turn printed LLVM text into **actual machine code you can call from Python**, completing the leap from tracing to a real just-in-time (JIT) runtime. We first bring in `llvmlite.binding` to access the low-level MCJIT interface, set up a native `execution_engine`, and add a helper that parses each generated module, links it into the engine, and finalises it. Inside `Variable.compile` we reuse the old `ir_module` builder, but immediately pass its string IR to `create_ir`, retrieve the function's address with `engine.get_function_address`, and wrap that raw pointer in a `ctypes.CFUNCTYPE` whose signature mirrors the LLVM function type. The result is a Python-callable object—`d_compiled`, `a_grad_compiled`, …—that dispatches straight into native code with zero per-call interpretation overhead. Running `d_compiled(1.0, 2.0, 3.0)` or the gradient functions now executes the compiled forward or backward kernel, giving us the same answers as before but at native speed and with the satisfying snap of real code generation—all built on fewer than a hundred lines of scaffolding.



```python
from llvmlite import ir
from ctypes import CFUNCTYPE, c_double
import llvmlite.binding as llvm

double = ir.DoubleType()

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

def create_execution_engine():
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    backing_mod = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
    return engine

def create_ir(engine, llvm_ir):
    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()
    return mod

engine = create_execution_engine()

class Input:
    def stringify(args):
        return str(args[0])

    def evaluate(args, inputs):
        return inputs[args[0]]

    def ir(args, builder, func, inputs_keys):
        return func.args[inputs_keys.index(args[0])]

class Add:
    def stringify(args):
        return f"({args[0]} + {args[1]})"

    def evaluate(args, inputs):
        return args[0].evaluate(inputs) + args[1].evaluate(inputs)

    def ir(args, builder, func, inputs_keys):
        lhs = args[0].ir(builder, func, inputs_keys)
        rhs = args[1].ir(builder, func, inputs_keys)
        return builder.fadd(lhs, rhs, name=Add.stringify(args))

class Sub:
    def stringify(args):
        return f"({args[0]} - {args[1]})"

    def evaluate(args, inputs):
        return args[0].evaluate(inputs) - args[1].evaluate(inputs)

    def ir(args, builder, func, inputs_keys):
        lhs = args[0].ir(builder, func, inputs_keys)
        rhs = args[1].ir(builder, func, inputs_keys)
        return builder.fsub(lhs, rhs, name=Sub.stringify(args))

class Mul:
    def stringify(args):
        return f"({args[0]} * {args[1]})"

    def evaluate(args, inputs):
        return args[0].evaluate(inputs) * args[1].evaluate(inputs)

    def ir(args, builder, func, inputs_keys):
        lhs = args[0].ir(builder, func, inputs_keys)
        rhs = args[1].ir(builder, func, inputs_keys)
        return builder.fmul(lhs, rhs, name=Mul.stringify(args))

class Div:
    def stringify(args):
        return f"({args[0]} / {args[1]})"

    def evaluate(args, inputs):
        return args[0].evaluate(inputs) / args[1].evaluate(inputs)

    def ir(args, builder, func, inputs_keys):
        lhs = args[0].ir(builder, func, inputs_keys)
        rhs = args[1].ir(builder, func, inputs_keys)
        return builder.fdiv(lhs, rhs, name=Div.stringify(args))

class Neg:
    def stringify(args):
        return f"(-{args[0]})"

    def evaluate(args, inputs):
        return -args[0].evaluate(inputs)

    def ir(args, builder, func, inputs_keys):
        return builder.fneg(args[0].ir(builder, func, inputs_keys), name=Neg.stringify(args))

class Variable:

    def __init__(self, op, args, grad_fn=None, grad=None):
        self.op = op
        self.args = args
        self.grad_fn = grad_fn
        self.grad = grad

    def __repr__(self):
        return self.op.stringify(self.args)

    def __str__(self):
        return self.op.stringify(self.args)

    def __add__(self, other: "Variable") -> "Variable":
        return Variable(Add, [self, other], grad_fn=lambda grad: [grad, grad])

    def __sub__(self, other: "Variable") -> "Variable":
        return Variable(Sub, [self, other], grad_fn=lambda grad: [grad, -grad])

    def __mul__(self, other: "Variable") -> "Variable":
        return Variable(Mul, [self, other], grad_fn=lambda grad: [grad * other, self * grad])

    def __truediv__(self, other: "Variable") -> "Variable":
        return Variable(Div, [self, other], grad_fn=lambda grad: [grad / other, -grad * self / other / other])

    def __neg__(self) -> "Variable":
        return Variable(Neg, [self], grad_fn=lambda grad: [-grad])

    def evaluate(self, inputs):
        return self.op.evaluate(self.args, inputs)

    def backward(self, grad):
        self.grad = grad if self.grad == None else self.grad + grad
        if self.op != Input:
            for p, g in zip(self.args, self.grad_fn(grad)):
                p.backward(g)

    def ir(self, builder, func, inputs_keys):
        return self.op.ir(self.args, builder, func, inputs_keys)

    def ir_module(self, func_name, return_type, func_signature):
        fnty = ir.FunctionType(return_type, func_signature.values())
        module = ir.Module(name=func_name)
        func = ir.Function(module, fnty, name=func_name)
        block = func.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)
        builder.ret(self.ir(builder, func, list(func_signature.keys())))
        return module

    def compile(self, func_name, return_type, func_signature):
        module = self.ir_module(func_name, return_type, func_signature)
        create_ir(engine, str(module))
        func_ptr = engine.get_function_address(func_name)
        return_type = c_double if return_type == double else None
        arg_types = [c_double if type == double else None for type in func_signature.values()]
        return CFUNCTYPE(return_type, *arg_types)(func_ptr)

a = Variable(Input, ['a'])
b = Variable(Input, ['b'])
c = Variable(Input, ['c'])
d = -a + b * (c - a / b)
d_compiled = d.compile('d', double, {'a': double, 'b': double, 'c': double})

d.backward(Variable(Input, ['d_grad']))
a_grad_compiled = a.grad.compile('a_grad', double, {'a': double, 'b': double, 'c': double, 'd_grad': double})
b_grad_compiled = b.grad.compile('b_grad', double, {'a': double, 'b': double, 'c': double, 'd_grad': double})
c_grad_compiled = c.grad.compile('c_grad', double, {'a': double, 'b': double, 'c': double, 'd_grad': double})

print(f"{d_compiled(1.0, 2.0, 3.0)=}")
print(f"{a_grad_compiled(1.0, 2.0, 3.0, 1.0)=}")
print(f"{b_grad_compiled(1.0, 2.0, 3.0, 1.0)=}")
print(f"{c_grad_compiled(1.0, 2.0, 3.0, 1.0)=}")
```

    d_compiled(1.0, 2.0, 3.0)=4.0
    a_grad_compiled(1.0, 2.0, 3.0, 1.0)=-2.0
    b_grad_compiled(1.0, 2.0, 3.0, 1.0)=3.0
    c_grad_compiled(1.0, 2.0, 3.0, 1.0)=2.0


## 7. Benchmark

To close the loop, we measure whether all this machinery actually **buys us speed**. First we compile a beefier forward expression `d` and its three reverse-mode gradients into native functions (`d_compiled`, `a_grad_compiled`, …). Then we generate a million random quadruples `(a, b, c, d_grad)` with NumPy and time two tight loops: one that calls the compiled kernels, and another that runs the same forward and backward logic through our tree interpreter (`evaluate`). Because each compiled function is just a C pointer wrapped by `ctypes`, every invocation jumps straight into optimized machine code with zero Python arithmetic or recursion, while the interpreted path re-evaluates the symbolic tree at each step. The timer comparison—`compiled_time` versus `evaluated_time`, printed alongside their ratio—shows the raw payoff of the JIT: you should see the compiled version slash runtime by an order of magnitude or more on this scalar workload, offering concrete evidence that the incremental steps from eager autograd to symbolic tracing to LLVM codegen weren't just academic—they translate into real wall-clock wins.



```python
from llvmlite import ir
from ctypes import CFUNCTYPE, c_double
import llvmlite.binding as llvm

double = ir.DoubleType()

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

def create_execution_engine():
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    backing_mod = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
    return engine

def create_ir(engine, llvm_ir):
    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()
    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()
    return mod

engine = create_execution_engine()

class Input:
    def stringify(args):
        return str(args[0])

    def evaluate(args, inputs):
        return inputs[args[0]]

    def ir(args, builder, func, inputs_keys):
        return func.args[inputs_keys.index(args[0])]

class Add:
    def stringify(args):
        return f"({args[0]} + {args[1]})"

    def evaluate(args, inputs):
        return args[0].evaluate(inputs) + args[1].evaluate(inputs)

    def ir(args, builder, func, inputs_keys):
        lhs = args[0].ir(builder, func, inputs_keys)
        rhs = args[1].ir(builder, func, inputs_keys)
        return builder.fadd(lhs, rhs, name=Add.stringify(args))

class Sub:
    def stringify(args):
        return f"({args[0]} - {args[1]})"

    def evaluate(args, inputs):
        return args[0].evaluate(inputs) - args[1].evaluate(inputs)

    def ir(args, builder, func, inputs_keys):
        lhs = args[0].ir(builder, func, inputs_keys)
        rhs = args[1].ir(builder, func, inputs_keys)
        return builder.fsub(lhs, rhs, name=Sub.stringify(args))

class Mul:
    def stringify(args):
        return f"({args[0]} * {args[1]})"

    def evaluate(args, inputs):
        return args[0].evaluate(inputs) * args[1].evaluate(inputs)

    def ir(args, builder, func, inputs_keys):
        lhs = args[0].ir(builder, func, inputs_keys)
        rhs = args[1].ir(builder, func, inputs_keys)
        return builder.fmul(lhs, rhs, name=Mul.stringify(args))

class Div:
    def stringify(args):
        return f"({args[0]} / {args[1]})"

    def evaluate(args, inputs):
        return args[0].evaluate(inputs) / args[1].evaluate(inputs)

    def ir(args, builder, func, inputs_keys):
        lhs = args[0].ir(builder, func, inputs_keys)
        rhs = args[1].ir(builder, func, inputs_keys)
        return builder.fdiv(lhs, rhs, name=Div.stringify(args))

class Neg:
    def stringify(args):
        return f"(-{args[0]})"

    def evaluate(args, inputs):
        return -args[0].evaluate(inputs)

    def ir(args, builder, func, inputs_keys):
        return builder.fneg(args[0].ir(builder, func, inputs_keys), name=Neg.stringify(args))

class Variable:

    def __init__(self, op, args, grad_fn=None, grad=None):
        self.op = op
        self.args = args
        self.grad_fn = grad_fn
        self.grad = grad

    def __repr__(self):
        return self.op.stringify(self.args)

    def __str__(self):
        return self.op.stringify(self.args)

    def __add__(self, other: "Variable") -> "Variable":
        return Variable(Add, [self, other], grad_fn=lambda grad: [grad, grad])

    def __sub__(self, other: "Variable") -> "Variable":
        return Variable(Sub, [self, other], grad_fn=lambda grad: [grad, -grad])

    def __mul__(self, other: "Variable") -> "Variable":
        return Variable(Mul, [self, other], grad_fn=lambda grad: [grad * other, self * grad])

    def __truediv__(self, other: "Variable") -> "Variable":
        return Variable(Div, [self, other], grad_fn=lambda grad: [grad / other, -grad * self / other / other])

    def __neg__(self) -> "Variable":
        return Variable(Neg, [self], grad_fn=lambda grad: [-grad])

    def evaluate(self, inputs):
        return self.op.evaluate(self.args, inputs)

    def backward(self, grad):
        self.grad = grad if self.grad == None else self.grad + grad
        if self.op != Input:
            for p, g in zip(self.args, self.grad_fn(grad)):
                p.backward(g)

    def ir(self, builder, func, inputs_keys):
        return self.op.ir(self.args, builder, func, inputs_keys)

    def ir_module(self, func_name, return_type, func_signature):
        fnty = ir.FunctionType(return_type, func_signature.values())
        module = ir.Module(name=func_name)
        func = ir.Function(module, fnty, name=func_name)
        block = func.append_basic_block(name="entry")
        builder = ir.IRBuilder(block)
        builder.ret(self.ir(builder, func, list(func_signature.keys())))
        return module

    def compile(self, func_name, return_type, func_signature):
        module = self.ir_module(func_name, return_type, func_signature)
        create_ir(engine, str(module))
        func_ptr = engine.get_function_address(func_name)
        return_type = c_double if return_type == double else None
        arg_types = [c_double if type == double else None for type in func_signature.values()]
        return CFUNCTYPE(return_type, *arg_types)(func_ptr)

a = Variable(Input, ['a'])
b = Variable(Input, ['b'])
c = Variable(Input, ['c'])
d = -a * b + (c - a / b) * (a + b) * c - (b * c - a) / (a + c) + b * b * c / a

d_compiled = d.compile('d', double, {'a': double, 'b': double, 'c': double})

d.backward(Variable(Input, ['d_grad']))

a_grad_compiled = a.grad.compile('a_grad', double, {'a': double, 'b': double, 'c': double, 'd_grad': double})
b_grad_compiled = b.grad.compile('b_grad', double, {'a': double, 'b': double, 'c': double, 'd_grad': double})
c_grad_compiled = c.grad.compile('c_grad', double, {'a': double, 'b': double, 'c': double, 'd_grad': double})

import time
import numpy as np

# Generate random input data
n_samples = 1000000
a_vals = np.random.randn(n_samples)
b_vals = np.random.randn(n_samples)
c_vals = np.random.randn(n_samples)
d_grad_vals = np.random.randn(n_samples)

# Benchmark compiled forward and backward pass
start = time.time()
for i in range(n_samples):
    d_compiled(a_vals[i], b_vals[i], c_vals[i])
    a_grad_compiled(a_vals[i], b_vals[i], c_vals[i], d_grad_vals[i])
    b_grad_compiled(a_vals[i], b_vals[i], c_vals[i], d_grad_vals[i])
    c_grad_compiled(a_vals[i], b_vals[i], c_vals[i], d_grad_vals[i])
compiled_time = time.time() - start

# Benchmark evaluated forward and backward pass
start = time.time()
for i in range(n_samples):
    inputs = {'a': a_vals[i], 'b': b_vals[i], 'c': c_vals[i], 'd_grad': d_grad_vals[i]}
    d.evaluate(inputs)
    a.grad.evaluate(inputs)
    b.grad.evaluate(inputs)
    c.grad.evaluate(inputs)
evaluated_time = time.time() - start

print(f"Compiled time: {compiled_time:.3f}s")
print(f"Evaluated time: {evaluated_time:.3f}s")
print(f"Speedup: {evaluated_time/compiled_time:.1f}x")
```

    Compiled time: 7.738s
    Evaluated time: 48.984s
    Speedup: 6.3x


The numbers tell a clear story: pushing the computation through our LLVM JIT trims a million forward-plus-backward evaluations from ≈48.984 seconds down to ≈7.738s seconds—a **6.3× speed-up**. Even for this scalar-only toy example, eliminating Python's per-operation overhead and recursive tree walks pays off handsomely; in higher-dimensional tensor settings (where LLVM can better exploit vectorization and common-sub-expression elimination) the gap would widen further.

