gpt-5 mini via github copilot

## Q:  hello. python. any tips on building a pytorch-like library from zero / scratch?


GitHub Copilot Chat Assistant — concise plan and practical tips.

High-level goals to decide first
- Educational toy vs production-grade (performance, CUDA, distributed). Start with educational — add complexity incrementally.
- API style: NumPy-like eager tensors (PyTorch) or functional/graph (JAX). Eager is easier to implement and reason about.

Minimum viable components (MVP)
1. Tensor core
  - Wrap a contiguous NumPy array (or CuPy for GPU) and store shape, dtype, device.
  - Flags: requires_grad, grad (accumulated), grad_fn (creator), is_leaf.
  - Overload arithmetic and matmul to produce new Tensors and record ops.

2. Autograd engine
  - Build a simple dynamic graph: each operation returns a tensor with a backward function pointing to parent tensors.
  - On backward(), do a topological sort (or use a stack) to call backward functions in correct order, accumulating grads.
  - Support basic ops: add, sub, mul, div, matmul, sum, mean, reshape, transpose, indexing, elementwise ops.
  - Handle broadcasting, reduction grads, and shape handling carefully.

3. NN module abstraction
  - Module class that can register Parameter objects (special tensors with requires_grad=True).
  - Layers (Linear, Conv, Activation) implemented as Modules using Tensor ops.
  - forward method convention and state_dict / load_state_dict.

4. Optimizers
  - Simple SGD, SGD with momentum, Adam; operate on parameters and their .grad, include zero_grad().

5. Utilities
  - no_grad context manager and Tensor.detach().
  - Save/load model (state_dict), basic serialization.
  - Gradient checking utilities (finite differences) for debugging.

Suggested implementation roadmap (practical milestones)
- Week 1: Minimal Tensor around NumPy + small set of ops (+, *, matmul, sum) + autograd backward for these ops.
- Week 2: Module, Parameter, basic Linear layer, small training loop (SGD) and example (MNIST or small synthetic).
- Week 3: Add more ops, broadcasting support, detach/no_grad, in-place op checks, gradient checking tests.
- Week 4+: GPU support (CuPy first), complex layers, performance optimizations, memory pooling, basic distributed/optimizer improvements.

Autograd design notes (common pitfalls)
- In-place operations: they can overwrite values needed for backward — either forbid or track and raise error.
- Views vs copies: maintain correct gradient propagation for views (store base and view metadata).
- Accumulation: accumulating gradients across backward calls — ensure .grad is initialized/zeroed appropriately.
- Higher-order gradients: if you want grad of grad, ensure backward functions record metadata needed.

Performance & scaling
- For prototyping use NumPy (CPU) and CuPy (GPU) to avoid writing C/CUDA initially.
- For performance-critical paths, implement kernels in C++/CUDA and bind via pybind11 or use Triton/Numba.
- Avoid Python loops in ops — rely on vectorized BLAS/NumPy/CuPy operations.
- Consider memory pool to reduce allocation overhead (PyTorch uses caching allocators).
- Fuse kernels (e.g., fused GEMM + bias + activation) for speed in production.

Practical engineering advice
- Keep a small, well-tested core. Tests: unit tests for ops, gradient checks, integration tests with training loops.
- Start with single-threaded correctness before parallelism.
- Use type annotations and small, clear APIs (Tensor, Function, Module, Optimizer).
- Maintain clear separation: frontend Python API vs backend execution (C++/CUDA later).
- Document the API and provide simple examples/tutorials.

Safety, licensing, and scope
- Reimplementing an entire PyTorch is huge — you’ll likely re-use ideas and maybe code references; respect licenses.
- Decide whether to interoperate with PyTorch/TorchScript or remain independent.

Resources & small references
- micrograd by Andrej Karpathy (tiny autograd educational impl)
- tinygrad (minimal GPU-capable framework)
- minitorch (educational)
- PyTorch source (torch.autograd, ATen, c10) when you’re ready for deeper design
- Papers/blogs on autodiff, fused kernels, memory allocators

Small starter sketch (conceptual)
- Implement a Tensor wrapper holding data, requires_grad, grad_fn.
- Each op creates a node with a backward closure that computes grads for parents.
- backward() starts from loss tensor, sets grad to ones, walks the graph, calls each backward closure to propagate grads.

Would you like:
- a tiny runnable toy implementation ( ~100–200 lines) that demonstrates a minimal tensor + autograd? 
- or links to educational repos on GitHub so you can inspect actual code?




## Q: can you show a tiny toy implementation with tensor and autograd holding, data, requires_grd, grad_fn?

A: 

Here's a tiny, self-contained toy implementation of a Tensor with a simple dynamic autograd engine (holds data, requires_grad, grad, grad_fn, parents). Save the file and run it — the bottom contains small examples (elementwise multiply and a tiny linear regression step).

```python name=toy_autograd.py
import numpy as np
from contextlib import contextmanager

# Global switch for gradient recording (like torch.no_grad)
_ENABLE_GRAD = True

@contextmanager
def no_grad():
    global _ENABLE_GRAD
    old = _ENABLE_GRAD
    _ENABLE_GRAD = False
    try:
        yield
    finally:
        _ENABLE_GRAD = old

def ensure_array(x):
    if isinstance(x, Tensor):
        return x
    return Tensor(np.array(x))

def unbroadcast(grad, shape):
    # Reduce grad to the target shape (inverse of numpy broadcasting)
    if grad.shape == shape:
        return grad
    # sum extra leading dimensions
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # sum over axes where shape has dimension 1
    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)

class Tensor:
    def __init__(self, data, requires_grad=False):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data
        self.requires_grad = bool(requires_grad) and _ENABLE_GRAD
        self.grad = None
        self._grad_fn = None   # callable that populates parent.grad using this.grad
        self._parents = []     # parent Tensors
        # simple flag to identify leaf tensors (no grad_fn)
        self.is_leaf = True

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    # ----- basic ops -----
    def _apply_op(self, other, op, grad_fn):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        data = op(self.data, other.data)
        requires_grad = (self.requires_grad or other.requires_grad) and _ENABLE_GRAD
        out = Tensor(data, requires_grad=requires_grad)
        if requires_grad:
            out.is_leaf = False
            out._parents = [self, other]
            def _backward():
                # closure sees out, self, other
                if out.grad is None:
                    return
                grad = out.grad
                if self.requires_grad:
                    ga = grad_fn(grad, self.data, other.data, which='a')
                    ga = unbroadcast(ga, self.data.shape)
                    self.grad = self.grad + ga if self.grad is not None else ga.copy()
                if other.requires_grad:
                    gb = grad_fn(grad, self.data, other.data, which='b')
                    gb = unbroadcast(gb, other.data.shape)
                    other.grad = other.grad + gb if other.grad is not None else gb.copy()
            out._grad_fn = _backward
        return out

    def __add__(self, other):
        return self._apply_op(other,
                              op=lambda a,b: a + b,
                              grad_fn=lambda grad, a, b, which:
                                  grad if which == 'a' else grad)

    __radd__ = __add__

    def __mul__(self, other):
        return self._apply_op(other,
                              op=lambda a,b: a * b,
                              grad_fn=lambda grad, a, b, which:
                                  grad * b if which == 'a' else grad * a)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad and _ENABLE_GRAD)
        if out.requires_grad:
            out.is_leaf = False
            out._parents = [self]
            def _backward():
                if out.grad is None:
                    return
                g = -out.grad
                self.grad = self.grad + g if self.grad is not None else g.copy()
            out._grad_fn = _backward
        return out

    def __sub__(self, other):
        return self + (-ensure_array(other))

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        data = self.data @ other.data
        requires_grad = (self.requires_grad or other.requires_grad) and _ENABLE_GRAD
        out = Tensor(data, requires_grad=requires_grad)
        if requires_grad:
            out.is_leaf = False
            out._parents = [self, other]
            def _backward():
                if out.grad is None:
                    return
                g = out.grad
                if self.requires_grad:
                    ga = g @ other.data.T
                    self.grad = self.grad + ga if self.grad is not None else ga.copy()
                if other.requires_grad:
                    gb = self.data.T @ g
                    other.grad = other.grad + gb if other.grad is not None else gb.copy()
            out._grad_fn = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(data, requires_grad=self.requires_grad and _ENABLE_GRAD)
        if out.requires_grad:
            out.is_leaf = False
            out._parents = [self]
            def _backward():
                if out.grad is None:
                    return
                g = out.grad
                if axis is None:
                    ga = np.ones_like(self.data) * g
                else:
                    # make grad have same dims if keepdims is False
                    grad = g
                    if not keepdims:
                        if isinstance(axis, int):
                            axes = (axis,)
                        else:
                            axes = tuple(axis)
                        for ax in sorted(axes):
                            grad = np.expand_dims(grad, ax)
                    ga = np.ones_like(self.data) * grad
                self.grad = self.grad + ga if self.grad is not None else ga.copy()
            out._grad_fn = _backward
        return out

    @property
    def T(self):
        data = self.data.T
        out = Tensor(data, requires_grad=self.requires_grad and _ENABLE_GRAD)
        if out.requires_grad:
            out.is_leaf = False
            out._parents = [self]
            def _backward():
                if out.grad is None:
                    return
                g = out.grad.T
                self.grad = self.grad + g if self.grad is not None else g.copy()
            out._grad_fn = _backward
        return out

    def reshape(self, *shape):
        data = self.data.reshape(*shape)
        out = Tensor(data, requires_grad=self.requires_grad and _ENABLE_GRAD)
        if out.requires_grad:
            out.is_leaf = False
            out._parents = [self]
            orig_shape = self.data.shape
            def _backward():
                if out.grad is None:
                    return
                g = out.grad.reshape(orig_shape)
                self.grad = self.grad + g if self.grad is not None else g.copy()
            out._grad_fn = _backward
        return out

    def detach(self):
        return Tensor(self.data.copy(), requires_grad=False)

    def zero_grad(self):
        self.grad = None

    # ----- autograd traversal -----
    def _build_topo(self, visited, topo):
        if id(self) in visited:
            return
        visited.add(id(self))
        for p in getattr(self, "_parents", []):
            p._build_topo(visited, topo)
        topo.append(self)

    def backward(self, gradient=None):
        # Only allow implicit gradient for scalars
        if gradient is None:
            if self.data.size != 1:
                raise RuntimeError("gradient must be specified for non-scalar tensors")
            gradient = np.ones_like(self.data)
        # initialize grads and topo
        self.grad = gradient if isinstance(gradient, np.ndarray) else np.array(gradient)
        topo = []
        visited = set()
        self._build_topo(visited, topo)
        # traverse in reverse topo and call grad fns
        for node in reversed(topo):
            if node._grad_fn is not None:
                node._grad_fn()

# ----------------- examples -----------------
if __name__ == "__main__":
    # Example 1: elementwise multiply and sum
    a = Tensor(np.array([2.0, 3.0]), requires_grad=True)
    b = Tensor(np.array([4.0, 5.0]), requires_grad=True)
    c = (a * b).sum()   # scalar
    c.backward()
    print("Example 1")
    print("a.grad:", a.grad)  # should be b.data -> [4,5]
    print("b.grad:", b.grad)  # should be a.data -> [2,3] 

    # Example 2: tiny linear regression step (one step of SGD)
    np.random.seed(0)
    X = np.random.randn(5, 3)
    true_w = np.array([1.5, -2.0, 0.5])
    y = X @ true_w + 0.1 * np.random.randn(5)

    w = Tensor(np.zeros(3), requires_grad=True)
    b = Tensor(0.0, requires_grad=True)
    lr = 0.1

    # forward
    preds = Tensor(X) @ w + b        # shape (5,)
    residuals = preds - Tensor(y)    # shape (5,)
    loss = (residuals * residuals).sum() * (1.0 / X.shape[0])  # MSE

    # backward
    w.zero_grad(); b.zero_grad()
    loss.backward()
    print("\nExample 2")
    print("loss:", loss.data)
    print("w.grad:", w.grad)  # gradient estimate
    print("b.grad:", b.grad)

    # simple SGD step
    w.data -= lr * w.grad
    b.data -= lr * b.grad
    print("w after one SGD step:", w.data)
```

What I provided and why
- A single-file minimal framework showing:
  - Tensor stores numpy-backed data, requires_grad, grad.
  - Operations create output Tensors with closures capturing backward logic.
  - A simple dynamic graph traversal (topological ordering) runs backward functions to accumulate gradients.
  - Basic ops: add, mul, matmul, neg, sub, sum, transpose, reshape.
  - Broadcasting-aware gradient reduction (unbroadcast).
  - no_grad context manager and detach.
  - Two runnable examples (elementwise gradient check and a tiny linear regression update).

Notes and next steps you might want
- This is intentionally small and not production-ready: it doesn't guard in-place ops, handle views, advanced broadcasting shapes, or higher-order gradients.
- If you want, I can:
  - Extend ops (divide, exp, relu, log, etc.).
  - Implement grad-check (finite differences) tests.
  - Add Parameter/Module and a simple optimizer class.
  - Show how to switch to GPU with CuPy (minimal changes).


  o o o

## Q: can you show how to add relu, dropout and softmax?



A:




 ````python name=toy_autograd_with_relu_dropout_softmax.py
# Tiny autograd toy with ReLU, Dropout, Softmax
# Self-contained single-file example (NumPy-backed).
# Run this file to see small tests at the bottom.

import numpy as np
from contextlib import contextmanager

# Global switch for gradient recording (like torch.no_grad)
_ENABLE_GRAD = True

@contextmanager
def no_grad():
    global _ENABLE_GRAD
    old = _ENABLE_GRAD
    _ENABLE_GRAD = False
    try:
        yield
    finally:
        _ENABLE_GRAD = old

def ensure_array(x):
    if isinstance(x, Tensor):
        return x
    return Tensor(np.array(x))

def unbroadcast(grad, shape):
    # Reduce grad to the target shape (inverse of numpy broadcasting)
    if grad.shape == shape:
        return grad
    # sum extra leading dimensions
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # sum over axes where shape has dimension 1
    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)

class Tensor:
    def __init__(self, data, requires_grad=False):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data
        self.requires_grad = bool(requires_grad) and _ENABLE_GRAD
        self.grad = None
        self._grad_fn = None   # callable that populates parent.grad using this.grad
        self._parents = []     # parent Tensors
        self.is_leaf = True

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    # ----- basic ops -----
    def _apply_op(self, other, op, grad_fn):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        data = op(self.data, other.data)
        requires_grad = (self.requires_grad or other.requires_grad) and _ENABLE_GRAD
        out = Tensor(data, requires_grad=requires_grad)
        if requires_grad:
            out.is_leaf = False
            out._parents = [self, other]
            def _backward():
                if out.grad is None:
                    return
                grad = out.grad
                if self.requires_grad:
                    ga = grad_fn(grad, self.data, other.data, which='a')
                    ga = unbroadcast(ga, self.data.shape)
                    self.grad = self.grad + ga if self.grad is not None else ga.copy()
                if other.requires_grad:
                    gb = grad_fn(grad, self.data, other.data, which='b')
                    gb = unbroadcast(gb, other.data.shape)
                    other.grad = other.grad + gb if other.grad is not None else gb.copy()
            out._grad_fn = _backward
        return out

    def __add__(self, other):
        return self._apply_op(other,
                              op=lambda a,b: a + b,
                              grad_fn=lambda grad, a, b, which:
                                  grad if which == 'a' else grad)

    __radd__ = __add__

    def __mul__(self, other):
        return self._apply_op(other,
                              op=lambda a,b: a * b,
                              grad_fn=lambda grad, a, b, which:
                                  grad * b if which == 'a' else grad * a)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad and _ENABLE_GRAD)
        if out.requires_grad:
            out.is_leaf = False
            out._parents = [self]
            def _backward():
                if out.grad is None:
                    return
                g = -out.grad
                self.grad = self.grad + g if self.grad is not None else g.copy()
            out._grad_fn = _backward
        return out

    def __sub__(self, other):
        return self + (-ensure_array(other))

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        data = self.data @ other.data
        requires_grad = (self.requires_grad or other.requires_grad) and _ENABLE_GRAD
        out = Tensor(data, requires_grad=requires_grad)
        if requires_grad:
            out.is_leaf = False
            out._parents = [self, other]
            def _backward():
                if out.grad is None:
                    return
                g = out.grad
                if self.requires_grad:
                    ga = g @ other.data.T
                    self.grad = self.grad + ga if self.grad is not None else ga.copy()
                if other.requires_grad:
                    gb = self.data.T @ g
                    other.grad = other.grad + gb if other.grad is not None else gb.copy()
            out._grad_fn = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(data, requires_grad=self.requires_grad and _ENABLE_GRAD)
        if out.requires_grad:
            out.is_leaf = False
            out._parents = [self]
            def _backward():
                if out.grad is None:
                    return
                g = out.grad
                if axis is None:
                    ga = np.ones_like(self.data) * g
                else:
                    grad = g
                    if not keepdims:
                        if isinstance(axis, int):
                            axes = (axis,)
                        else:
                            axes = tuple(axis)
                        for ax in sorted(axes):
                            grad = np.expand_dims(grad, ax)
                    ga = np.ones_like(self.data) * grad
                self.grad = self.grad + ga if self.grad is not None else ga.copy()
            out._grad_fn = _backward
        return out

    @property
    def T(self):
        data = self.data.T
        out = Tensor(data, requires_grad=self.requires_grad and _ENABLE_GRAD)
        if out.requires_grad:
            out.is_leaf = False
            out._parents = [self]
            def _backward():
                if out.grad is None:
                    return
                g = out.grad.T
                self.grad = self.grad + g if self.grad is not None else g.copy()
            out._grad_fn = _backward
        return out

    def reshape(self, *shape):
        data = self.data.reshape(*shape)
        out = Tensor(data, requires_grad=self.requires_grad and _ENABLE_GRAD)
        if out.requires_grad:
            out.is_leaf = False
            out._parents = [self]
            orig_shape = self.data.shape
            def _backward():
                if out.grad is None:
                    return
                g = out.grad.reshape(orig_shape)
                self.grad = self.grad + g if self.grad is not None else g.copy()
            out._grad_fn = _backward
        return out

    def detach(self):
        return Tensor(self.data.copy(), requires_grad=False)

    def zero_grad(self):
        self.grad = None

    # ----- new ops: relu, dropout, softmax -----
    def relu(self):
        data = np.maximum(0, self.data)
        out = Tensor(data, requires_grad=self.requires_grad and _ENABLE_GRAD)
        if out.requires_grad:
            out.is_leaf = False
            out._parents = [self]
            mask = (self.data > 0).astype(self.data.dtype)
            def _backward():
                if out.grad is None:
                    return
                g = out.grad * mask
                self.grad = self.grad + g if self.grad is not None else g.copy()
            out._grad_fn = _backward
        return out

    def dropout(self, p=0.5, training=True, seed=None):
        """
        Inverted dropout: during training, output = input * mask / (1-p)
        During eval (training=False) or inside no_grad, returns input (identity).
        """
        if not training or not _ENABLE_GRAD or p == 0.0:
            # No randomness / no effect; keep requires_grad same as input (but disabled if _ENABLE_GRAD False)
            return Tensor(self.data.copy(), requires_grad=self.requires_grad and _ENABLE_GRAD)

        if p < 0.0 or p >= 1.0:
            raise ValueError("p must be in [0, 1) for dropout")

        # determinism if seed provided
        rng = np.random if seed is None else np.random.RandomState(seed)

        mask = (rng.rand(*self.data.shape) >= p).astype(self.data.dtype)
        scale = 1.0 / (1.0 - p)
        data = self.data * mask * scale
        out = Tensor(data, requires_grad=self.requires_grad and _ENABLE_GRAD)
        if out.requires_grad:
            out.is_leaf = False
            out._parents = [self]
            # capture mask and scale for backward
            def _backward():
                if out.grad is None:
                    return
                g = out.grad * mask * scale
                self.grad = self.grad + g if self.grad is not None else g.copy()
            out._grad_fn = _backward
        return out

    def softmax(self, axis=-1):
        # Numerically stable softmax
        x = self.data
        # subtract max for stability
        x_max = np.max(x, axis=axis, keepdims=True)
        exps = np.exp(x - x_max)
        s = exps / np.sum(exps, axis=axis, keepdims=True)
        out = Tensor(s, requires_grad=self.requires_grad and _ENABLE_GRAD)
        if out.requires_grad:
            out.is_leaf = False
            out._parents = [self]
            def _backward():
                if out.grad is None:
                    return
                g = out.grad
                # For each slice along axis, grad_input = s * (g - sum(g * s, axis))
                dot = np.sum(g * s, axis=axis, keepdims=True)
                gi = s * (g - dot)
                self.grad = self.grad + gi if self.grad is not None else gi.copy()
            out._grad_fn = _backward
        return out

    # ----- autograd traversal -----
    def _build_topo(self, visited, topo):
        if id(self) in visited:
            return
        visited.add(id(self))
        for p in getattr(self, "_parents", []):
            p._build_topo(visited, topo)
        topo.append(self)

    def backward(self, gradient=None):
        # Only allow implicit gradient for scalars
        if gradient is None:
            if self.data.size != 1:
                raise RuntimeError("gradient must be specified for non-scalar tensors")
            gradient = np.ones_like(self.data)
        # initialize grads and topo
        self.grad = gradient if isinstance(gradient, np.ndarray) else np.array(gradient)
        topo = []
        visited = set()
        self._build_topo(visited, topo)
        # traverse in reverse topo and call grad fns
        for node in reversed(topo):
            if node._grad_fn is not None:
                node._grad_fn()

# ----------------- examples / tests -----------------
if __name__ == "__main__":
    np.random.seed(0)

    print("RELU test")
    x = Tensor(np.array([-1.0, 0.5, 2.0]), requires_grad=True)
    y = x.relu().sum()
    y.backward()
    print("x.data:", x.data)
    print("y:", y.data)
    print("x.grad (should be [0,1,1]):", x.grad)

    print("\nDROPOUT test (training=True)")
    x = Tensor(np.ones((4,)), requires_grad=True)
    # fixed seed so mask is deterministic for the example
    out = x.dropout(p=0.5, training=True, seed=42)
    print("dropout mask applied (out.data):", out.data)
    loss = out.sum()
    x.zero_grad()
    loss.backward()
    print("x.grad (should equal mask * scale):", x.grad)

    print("\nDROPOUT test (training=False / eval mode or inside no_grad)")
    x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    out_eval = x.dropout(p=0.5, training=False)
    print("out_eval.data (should equal input):", out_eval.data)
    # gradient should flow normally through identity
    out_eval.sum().backward()
    print("x.grad (should be ones):", x.grad)

    print("\nSOFTMAX test")
    logits = Tensor(np.array([[1.0, 2.0, 3.0],
                              [0.1, 0.2, 0.3]]), requires_grad=True)
    probs = logits.softmax(axis=1)
    # pretend loss is sum(probs * some grad)
    upstream = np.array([[0.1, 0.2, -0.3],
                         [0.0, 1.0, -1.0]])
    probs.grad = upstream
    # run backward manually on probs to propagate to logits
    probs._grad_fn()  # we can call grad_fn directly because we set probs.grad
    print("probs.data:\n", probs.data)
    print("logits.grad (from softmax jacobian-vector product):\n", logits.grad)
````


What I added and why
- Implemented relu(), dropout(p, training=True, seed=None), and softmax(axis=-1) methods on the Tensor class.
- Each operation builds an output Tensor and — when gradients are enabled — a backward closure capturing the necessary intermediate (mask for dropout, activation mask for ReLU, softmax outputs for softmax).
- Dropout uses inverted dropout scaling (divide by 1-p) and accepts a deterministic seed for reproducible masks in examples. When training=False (or inside no_grad), dropout is identity.
- Softmax backward uses the efficient vector-Jacobian product formula: grad_input = s * (grad - sum(grad * s)).

Notes and next steps
- This toy code is for learning and debugging; it doesn't cover in-place ops, view semantics, higher-order gradients, or performance concerns.
- If you want, I can:
  - Add unit tests (including finite-difference gradient checks) for these ops.
  - Add a small Module/Parameter + optimizer example using ReLU and Dropout in a tiny network.
  - Show how to implement CrossEntropy (log-softmax + NLL) with a numerically stable backward.






