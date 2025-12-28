"""
Tiny toy of PyTorch-like Module mechanics, extended with:
- Sequence: a Sequential-like container that registers its children
- Dropout: NumPy-based inverted dropout module (uses self.training)

Run this file to see registration, state_dict, and Dropout train/eval behavior.
"""
from collections import OrderedDict
import numpy as np
from typing import Dict, Iterator, Tuple, Any, Iterable


class Parameter:
    """Tiny Parameter wrapper around a numpy array (or any value)."""
    def __init__(self, data):
        self.data = np.asarray(data)

    def __repr__(self):
        return f"Parameter(shape={self.data.shape}, dtype={self.data.dtype})"


class Module:
    """Minimal Module supporting auto-registration of submodules/parameters/buffers."""
    def __init__(self):
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "training", True)

    # --- Registration helpers ------------------------------------------------
    def add_module(self, name: str, module: "Module"):
        if module is None:
            raise ValueError("module is None")
        self._modules[name] = module

    def register_parameter(self, name: str, param: Parameter):
        if param is None:
            if name in self._parameters:
                del self._parameters[name]
            return
        self._parameters[name] = param

    def register_buffer(self, name: str, buf):
        if buf is None:
            if name in self._buffers:
                del self._buffers[name]
            return
        self._buffers[name] = buf

    # --- Attribute interception ---------------------------------------------
    def __setattr__(self, name: str, value: Any):
        # keep private attributes untouched
        if name.startswith("_"):
            return object.__setattr__(self, name, value)

        # Module -> register as submodule
        if isinstance(value, Module):
            self.add_module(name, value)
            return object.__setattr__(self, name, value)

        # Parameter -> register as parameter
        if isinstance(value, Parameter):
            self.register_parameter(name, value)
            return object.__setattr__(self, name, value)

        # Buffers (numpy arrays, etc.) — treat np.ndarray as buffer here
        if isinstance(value, np.ndarray):
            self.register_buffer(name, value)
            return object.__setattr__(self, name, value)

        # fallback: normal attribute
        return object.__setattr__(self, name, value)

    def __delattr__(self, name: str):
        if name in self._modules:
            del self._modules[name]
        if name in self._parameters:
            del self._parameters[name]
        if name in self._buffers:
            del self._buffers[name]
        object.__delattr__(self, name)

    # --- Traversal / inspection ---------------------------------------------
    def children(self) -> Iterator["Module"]:
        return iter(self._modules.values())

    def modules(self) -> Iterator["Module"]:
        yield self
        for m in self._modules.values():
            yield from m.modules()

    def parameters(self) -> Iterator[Parameter]:
        # yield own parameters then ones from children
        for p in self._parameters.values():
            yield p
        for child in self._modules.values():
            yield from child.parameters()

    def named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        # own named params
        for name, p in self._parameters.items():
            yield (prefix + name, p)
        # child params with dotted names
        for child_name, child in self._modules.items():
            child_prefix = prefix + child_name + "."
            yield from child.named_parameters(child_prefix)

    # --- mode switching (train/eval) ----------------------------------------
    def train(self, mode: bool = True):
        self.training = bool(mode)
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    # --- state dict ---------------------------------------------------------
    def state_dict(self, prefix: str = "") -> Dict[str, np.ndarray]:
        """
        Collect parameters and buffers into a flat dict with dotted names.
        Example keys: "fc.weight", "layer1.conv.bias"
        """
        sd: Dict[str, np.ndarray] = {}
        # parameters
        for name, param in self._parameters.items():
            sd[prefix + name] = param.data
        # buffers
        for name, buf in self._buffers.items():
            sd[prefix + name] = buf
        # children
        for child_name, child in self._modules.items():
            child_prefix = prefix + child_name + "."
            sd.update(child.state_dict(child_prefix))
        return sd

    def load_state_dict(self, state: Dict[str, np.ndarray]):
        """
        Very simple load: set .data for parameters/buffers found in state.
        """
        for key, val in state.items():
            parts = key.split(".")
            obj = self
            for name in parts[:-1]:
                if name in obj._modules:
                    obj = obj._modules[name]
                else:
                    raise KeyError(f"Module path not found while loading key '{key}'")
            final = parts[-1]
            if final in obj._parameters:
                obj._parameters[final].data = np.asarray(val)
            elif final in obj._buffers:
                obj._buffers[final] = np.asarray(val)
            else:
                raise KeyError(f"Parameter/Buffer '{final}' not found in module when loading key '{key}'")

    def __repr__(self):
        cls = self.__class__.__name__
        mods = ", ".join(self._modules.keys())
        params = ", ".join(self._parameters.keys())
        return f"<{cls} modules=[{mods}] params=[{params}]>"


class ModuleList(Module):
    """
    Simple list container that registers added modules.
    Supports append and __getitem__ for demo purposes.
    """
    def __init__(self, modules: Iterable[Module] = None):
        super().__init__()
        self._list = []
        if modules:
            for m in modules:
                self.append(m)

    def append(self, module: Module):
        if not isinstance(module, Module):
            raise TypeError("ModuleList only accepts Module instances")
        name = str(len(self._list))
        # register under numeric string name
        self.add_module(name, module)
        self._list.append(module)

    def __getitem__(self, idx):
        return self._list[idx]

    def __len__(self):
        return len(self._list)

    def __repr__(self):
        return f"ModuleList({self._list})"


# ----------------- Sequence (Sequential-like) --------------------------------
class Sequence(Module):
    """
    Sequential-like container. Accepts varargs or a single iterable.
    Registers children under "0", "1", ... preserving order.
    """
    def __init__(self, *layers: Module):
        super().__init__()
        # allow Sequence([a,b,c]) or Sequence(a,b,c)
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            layers = layers[0]
        for i, layer in enumerate(layers):
            if not isinstance(layer, Module):
                raise TypeError("Sequence only accepts Module instances")
            self.add_module(str(i), layer)

    def forward(self, x):
        # chain children in insertion order
        for child in self._modules.values():
            x = child(x)
        return x

    def __len__(self):
        return len(self._modules)

    def __getitem__(self, idx):
        vals = list(self._modules.values())
        if isinstance(idx, slice):
            # return a new Sequence with the sliced modules
            return Sequence(vals[idx])
        if isinstance(idx, int):
            return vals[idx]
        raise TypeError("Index must be int or slice")


# ----------------- Dropout (NumPy inverted dropout) ------------------------
class Dropout(Module):
    """
    Inverted Dropout implemented with numpy.

    - During training: randomly zeroes elements with probability p and scales
      the remaining elements by 1/(1-p) so the expected value is unchanged.
    - During eval: returns the input unchanged.

    Works with any array-like input.
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError("p must satisfy 0 <= p < 1")
        self.p = float(p)

    def forward(self, x):
        arr = np.asarray(x)
        # If input is integer dtype, convert to float for scaling
        if arr.dtype.kind in ("i", "u"):
            arr = arr.astype(np.float32)

        if (not getattr(self, "training", True)) or self.p == 0.0:
            return arr

        keep_prob = 1.0 - self.p
        # create mask of same shape: 1 with probability keep_prob, 0 otherwise
        mask = (np.random.rand(*arr.shape) < keep_prob).astype(arr.dtype)
        # inverted dropout scaling
        mask = mask / keep_prob
        return arr * mask

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


# ----------------- tiny Layers for demo ------------------------------------
class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # register weight and bias as Parameters via attribute assignment
        self.weight = Parameter(np.zeros((out_features, in_features), dtype=float))
        self.bias = Parameter(np.zeros((out_features,), dtype=float))

    def forward(self, x):
        # naive forward (x is numpy array shape (..., in_features))
        return x.dot(self.weight.data.T) + self.bias.data

    def __repr__(self):
        return f"Linear(weight_shape={self.weight.data.shape}, bias_shape={self.bias.data.shape})"


# ----------------- demo ----------------------------------------------------
if __name__ == "__main__":
    # build a toy model that uses Sequence and Dropout
    class Net(Module):
        def __init__(self):
            super().__init__()
            # registered as submodule by __setattr__
            self.seq = Sequence(
                Linear(4, 3),
                Dropout(p=0.5),
                Linear(3, 2),
            )

            # ModuleList example
            self.layers = ModuleList([Linear(2, 2), Linear(2, 2)])

            # plain python list: NOT registered
            self.plain_list = [Linear(2, 2)]

            # a buffer (e.g., running stats) -> registering by setting numpy array
            self.running_mean = np.zeros(2)

    net = Net()
    print("net._modules keys:", list(net._modules.keys()))
    print("seq._modules keys (children of Sequence):", list(net.seq._modules.keys()))
    print("plain_list is attribute but NOT registered as modules:", "plain_list" in net._modules)

    # list all named parameters with dotted names
    print("\nNamed parameters:")
    for name, p in net.named_parameters():
        print(" ", name, "->", p)

    # state_dict
    sd = net.state_dict()
    print("\nstate_dict keys:")
    for k in sd:
        print(" ", k, "shape:", sd[k].shape)

    # Demonstrate Dropout behavior
    np.random.seed(0)
    x = np.arange(8.).reshape(2, 4)
    print("\nInput:\n", x)

    # ensure training mode
    net.train()
    print("\nnet.training (should be True):", net.training)
    out_train = net.seq(x)
    print("Output through seq in training (dropout applied):\n", out_train)

    # switch to eval() and run again (dropout is no-op)
    net.eval()
    print("\nnet.training after eval() (should be False):", net.training)
    out_eval = net.seq(x)
    print("Output through seq in eval (dropout not applied):\n", out_eval)