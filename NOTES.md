#  More PyTorch (with Autograd) from Scratch / Zero Notes


## Micrograd Family

uses scalar Value (0D-Tensor) only

**Micrograd** <https://github.com/karpathy/micrograd>

**Nanograd** <https://github.com/AbdulmalikDS/nanograd>


## PyTorch Internals

<https://blog.ezyang.com/2019/05/pytorch-internals/>




## Build PyTorch from Scratch / Zero


**Nets** <https://github.com/arthurdjn/nets>
-  <https://github.com/arthurdjn/nets/blob/master/nets/tensor.py> 
-  <https://github.com/arthurdjn/nets/blob/master/nets/autograd/function.py>




**PyNorch** <https://github.com/lucasdelimanogueira/PyNorch>

Recreating PyTorch from Scratch (with GPU Support and Automatic Differentiation) by Lucas de Lima Nogueira
<https://medium.com/data-science/recreating-pytorch-from-scratch-with-gpu-support-and-automatic-differentiation-8f565122a3cc>

```
# norch/autograd/functions.py

class AddBackward:
    def __init__(self, x, y):
        self.input = [x, y]

    def backward(self, gradient):
        return [gradient, gradient]


class SinBackward:
    def __init__(self, x):
        self.input = [x]

    def backward(self, gradient):
        x = self.input[0]
        return [x.cos() * gradient]

class ElementwiseMulBackward:
    def __init__(self, x, y):
        self.input = [x, y]

    def backward(self, gradient):
        x = self.input[0]
        y = self.input[1]
        return [y * gradient, x * gradient]

class SumBackward:
    def __init__(self, x):
        self.input = [x]

    def backward(self, gradient):
        # Since sum reduces a tensor to a scalar,
        #  gradient is broadcasted to match the original shape.
        return [float(gradient.tensor.contents.data[0]) * self.input[0].ones_like()]



# norch/tensor.py

def __add__(self, other):
    
  if self.shape != other.shape:
      raise ValueError("Tensors must have the same shape for addition")
   
  result_data = Tensor()
  
  result_data.requires_grad = self.requires_grad or other.requires_grad
  if result_data.requires_grad:
      result_data.grad_fn = AddBackward(self, other)

def backward(self, gradient=None):
    if not self.requires_grad:
        return
    
    if gradient is None:
        if self.shape == [1]:
            gradient = Tensor([1]) # dx/dx = 1 case
        else:
            raise RuntimeError("Gradient argument must be specified for non-scalar tensors.")

    if self.grad is None:
        self.grad = gradient

    else:
        self.grad += gradient

    if self.grad_fn is not None: # not a leaf
        grads = self.grad_fn.backward(gradient) # call the operation backward
        for tensor, grad in zip(self.grad_fn.input, grads):
            if isinstance(tensor, Tensor):
                tensor.backward(grad) # recursively call the backward again
                                      # for the gradient expression (chain rule)


def zero_grad(self):
    self.grad = None

def detach(self):
    self.grad = None
    self.grad_fn = None


# norch/nn/module.py

from .parameter import Parameter
from collections import OrderedDict
from abc import ABC
import inspect

class Module(ABC):
    """
    Abstract class for modules
    """
    def __init__(self):
        self._modules = OrderedDict()
        self._params = OrderedDict()
        self._grads = OrderedDict()
        self.training = True

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def train(self):
        self.training = True
        for param in self.parameters():
            param.requires_grad = True

    def eval(self):
        self.training = False
        for param in self.parameters():
            param.requires_grad = False

    def parameters(self):
        for name, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield self, name, value
            elif isinstance(value, Module):
                yield from value.parameters()

    def modules(self):
        yield from self._modules.values()

    def gradients(self):
        for module in self.modules():
            yield module._grads

    def zero_grad(self):
        for _, _, parameter in self.parameters():
            parameter.zero_grad()

    def to(self, device):
        for _, _, parameter in self.parameters():
            parameter.to(device)

        return self
    
    def inner_repr(self):
        return ""

    def __repr__(self):
        string = f"{self.get_name()}("
        tab = "   "
        modules = self._modules
        if modules == {}:
            string += f'\n{tab}(parameters): {self.inner_repr()}'
        else:
            for key, module in modules.items():
                string += f"\n{tab}({key}): {module.get_name()}({module.inner_repr()})"
        return f'{string}\n)'
    
    def get_name(self):
        return self.__class__.__name__
    
    def __setattr__(self, key, value):
        self.__dict__[key] = value

        if isinstance(value, Module):
            self._modules[key] = value
        elif isinstance(value, Parameter):
            self._params[key] = value

# norch/nn/loss.py

from .module import Module
 
class MSELoss(Module):
    def __init__(self):
      pass

    def forward(self, predictions, labels):
        assert labels.shape == predictions.shape, \
            "Labels and predictions shape does not match: {} and {}".format(labels.shape, predictions.shape)
        
        return ((predictions - labels) ** 2).sum() / predictions.numel

    def __call__(self, *inputs):
        return self.forward(*inputs)            
```




**MyTorch** <https://github.com/tigert1998/mytorch> -
A toy Python DL training library with PyTorch like API






## Tiny Torch

 <https://mlsysbook.ai/tinytorch/intro.html>

> A Build-It-Yourself Companion to the Machine Learning Systems textbook
>
> Build Your Own ML Framework. Don't import it. Build it.
> From tensors to systems. An educational framework for building and
> optimizing ML—understand how PyTorch, TensorFlow, and JAX really work.


> TinyTorch takes you from basic tensors to production-ready ML systems through 20 progressive modules. Here’s how they connect:

> Three tiers, one complete system:

> Foundation (blue): Build the core machinery—tensors hold data, activations add non-linearity, layers combine them, losses measure error, autograd computes gradients, optimizers update weights, and training orchestrates the loop. Each piece answers “what do I need to learn next?”

> Architecture (purple): Apply your foundation to real problems. DataLoader feeds data efficiently, then you choose your path: Convolutions for images or Transformers for text (Tokenization → Embeddings → Attention → Transformers).

> Optimization (orange): Make it fast. Profile to find bottlenecks, then apply quantization, compression, acceleration, or memoization. Benchmarking measures your improvements.


```
class Adam:
   def __init__(self, params, lr=0.001):
     self.params = params
     self.lr = lr
     # 2× optimizer state:
     # momentum + variance
     self.m = [zeros_like(p) for p in params]
     self.v = [zeros_like(p) for p in params]
 
   def step(self):
     for i, p in enumerate(self.params):
       self.m[i] = 0.9*self.m[i] + 0.1*p.grad
       self.v[i] = 0.99*self.v[i] + 0.001* p.grad**2
     p.data -= self.lr * self.m[i] / (self.v[i].sqrt()+1e-8)
```

### Educational ML Frameworks**

Educational frameworks teaching ML internals occupy
different points in the scope-simplicity tradeoff space.

**micrograd (Karpathy, 2022)** demonstrates autograd mechanics elegantly in approximately 200 lines of scalarvalued Python, making backpropagation transparent
through decomposition into elementary operations. Its
pedagogical clarity comes from intentional minimalism:
scalar operations only, no tensor abstraction, focused
solely on automatic differentiation fundamentals. This
design illuminates gradient mechanics but necessarily
omits systems concerns (memory profiling, computational complexity, production patterns) and modern architectures.

**MiniTorch (Rush, 2020)** extends beyond autograd to
tensor operations, neural network modules, and optional
GPU programming, originating from Cornell Tech’s Machine Learning Engineering course. The curriculum progresses from foundational autodifferentiation through
deep learning with assessment infrastructure (unit tests,
visualization tools). While MiniTorch includes an optional GPU module exploring parallel programming concepts and covers efficiency considerations throughout,
the core curriculum emphasizes mathematical rigor: students work through detailed exercises building tensor abstractions from first principles. TinyTorch differs through
systems-first emphasis (memory profiling and complexity analysis embedded from Module 01), productioninspired package organization, and three integration
models supporting diverse deployment contexts.

**tinygrad (Messina et al.)** positions itself between micrograd’s simplicity and PyTorch’s production capabilities, providing a complete framework (tensor library,
IR, compiler, JIT) that emphasizes hackability and transparency. Unlike opaque production frameworks, tiny
grad makes “the entire compiler and IR visible,” enabling students to understand deep learning compilation internals. While pedagogically valuable through its
inspectable design, tinygrad assumes significant background: students must navigate compiler concepts, multiple hardware backends, and production-level architecture without scaffolded progression or automated assessment infrastructure.


Framework | Purpose | Scope | Systems Focus | Target Outcome
micrograd | Teach autograd |Autograd only (scalar) | Minimal | Understand backprop
MiniTorch | Teach ML math  | Tensors + autograd + optional GPU | Math foundations |  Build from first principles
tinygrad  | Inspectable production | Complete (compiler, IR, JIT) | Advanced (compiler) |  Understand compilation
TinyTorch  | Teach systems | Complete (tensors → transformers → optimization) | Embedded from Module 01 |  Framework engineers


Foundation Tier (01–08)
- 01 Fnd Tensor Multidimensional arrays, broadcasting Memory footprint (nbytes), dtype sizes, contiguous layout
- 02 Fnd Activations ReLU, Sigmoid, Tanh, GELU, Softmax Numerical stability (exp overflow),vectorization
- 03 Fnd Layers Linear, Xavier initialization Parameter vs activation memory, weight layout
- 04 Fnd Losses Cross-entropy, MSE, log-sum-exp trick Numerical stability (log(0)), gradient magnitude
- 05 Fnd DataLoader Dataset abstraction, batching, shuffling Iterator protocol, batch collation overhead
- 06 Fnd Autograd Computational graphs, chain rule, backprop Gradient memory (2× momentum, 3×Adam)
- 07 Fnd Optimizers SGD, Momentum, Adam, AdamW Optimizer state memory, in-place updates
- 08 Fnd Training Cosine scheduling, gradient clipping Peak memory lifecycle, checkpoint tradeoffs



> Broadcasting Rules:
> 1. Start from rightmost dimension
> 2. Dimensions must be equal OR one must be 1 OR one must be missing
> 3. Missing dimensions are assumed to be 1
>
> **Key Insight**: Broadcasting makes tensors of different shapes compatible by automatically expanding dimensions. This is crucial for batch processing where you often add a single bias vector to an entire batch of data.
>
> **Memory Efficiency**: Broadcasting doesn't actually create expanded copies in memory - NumPy computes results on-the-fly, saving memory.



