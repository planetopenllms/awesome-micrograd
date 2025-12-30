from .autograd import Tensor
import numpy as np


class Module():
    def __init__(self):
        self.params = list()  ## note: same as []
                              ##   rename to _paramaters (from params) - why? why not?         
        self._modules = {}  ## Dict[str, Module]
        self.training: bool = True  # default training mode is True
  
    def parameters(self):
        return self.params

    def forward(self, *args, **kwargs):   
        raise NotImplementedError
    
    ## forward/alias to forward  - just use __call__ = forward or such - why? why not?
    def __call__(self, *args, **kwargs):  
        return self.forward(*args, **kwargs)

    ##################
    ## add support for train() and eval() mode
    ##     required for dropout (and batchnorm in future)
    def add_module(self, name, module):
        self._modules[name] = module

    def children(self):
        # direct children (non-recursive)
        return self._modules.values()

    """
    def modules(self):
        # yields self and all submodules recursively (similar to torch.nn.Module.modules)
        yield self
        for m in self._modules.values():
            yield from m.modules()
    """

    def train(self, mode=True):
        """
        Set this module into training mode (True) or eval mode (False) and
        recursively apply to all submodules.
        Returns self to allow chaining.
        """
        self.training = bool(mode)
        for m in self.children():
            m.train(mode)
        return self

    def eval(self):
        "Switch to evaluation mode (training=False)."
        return self.train(False)



class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        ## uses he/kaiming()-init with normal dist
        ##  fix use Tensor.normal( 0, var? or std=  sqrt(2.0/n_inputs))
        ##    fix use he/kaiming_init helper or such
        ##     plus allow passing in of different weight_init methods a la pytorch
        W = np.random.randn(in_features, out_features) * np.sqrt(2.0/(in_features))
        self.weight = Tensor(W, requires_grad=True)
        ## todo - change to self.register_parameter !!!
        self.params.append(self.weight)
        
        if bias:
          self.bias = Tensor.zeros(out_features, requires_grad=True)
          ## todo - change to self.register_parameter !!
          self.params.append(self.bias)
        else:
          self.bias = None

    def forward(self, input):
        ## note: no "broadcast" used for bias; bias gets expanded for batch dim
        x = input.mm(self.weight)
        if self.bias is not None:
            x += self.bias.expand(dim=0,copies=len(input.data))
        return x 
    

class Sequential(Module):    
    def __init__(self, *layers):
        super().__init__()
        # convenience & backward compat - accept either multiple args or a single iterable
        # if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
        #    layers = layers[0]
        for i, layer in enumerate(layers):
            self.add_module(str(i), layer)

    def add(self, layer):
        self.add_module(str(len(self._modules)), layer )
        
    def forward(self, input):
        for layer in self._modules.values():
            input = layer.forward(input)
        return input
    
    def parameters(self):
        params = list()
        for layer in self._modules.values():
            params += layer.parameters()
        return params

    #####################
    ## add access-via []  e.g. layer[0].weight etc.
    def __getitem__(self, index): 
        values = list(self._modules.values())
        return values[index]  
    def __len__(self):  
        return len(self._modules)



class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        assert (0.0 <= p < 1.0), "p must satisfy 0 <= p < 1"
        self.p = float(p)

    def forward(self, input): 
        # No-op in evaluation mode or when p == 0
        if self.training == False or self.p == 0.0:
            return input

        return input.dropout(p=self.p)
 

###
# -- loss function layers
class MSELoss(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        ## todo - add mean() to sum-up and divide by batch dim - why? why not?
        return ((pred - target)*(pred - target)).sum(dim=0)

class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        return pred.cross_entropy(target)



    
###
# -- activation function layers
class Tanh(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return input.tanh()
    
class Sigmoid(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return input.sigmoid()
    
class ReLU(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return input.relu()    
    
