from .autograd import Tensor
import numpy as np


class Module():
    def __init__(self):
        self.params = list()  ## note: same as []
                              ##   rename to _paramaters (from params) - why? why not?         
    def parameters(self):
        return self.params

    ## forward/alias to forward  - just use __call__ = forward or such - why? why not?
    def __call__(self, *args, **kwargs):   
        return self.forward(*args, **kwargs)



class Linear(Module):
    def __init__(self, n_inputs, n_outputs, bias=True):
        super().__init__()
        ## uses he/kaiming()-init with normal dist
        ##  fix use Tensor.normal( 0, var? or std=  sqrt(2.0/n_inputs))
        ##    fix use he/kaiming_init helper or such
        ##     plus allow passing in of different weight_init methods a la pytorch
        W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0/(n_inputs))
        self.weight = Tensor(W, requires_grad=True)
        self.params.append(self.weight)
        
        if bias:
          self.bias = Tensor.zeros(n_outputs, requires_grad=True)
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
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            layers = layers[0]
        self.layers = list(layers)
    
    def add(self, layer):
        self.layers.append(layer)
        
    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def parameters(self):
        params = list()
        for l in self.layers:
            params += l.parameters()
        return params

    #####################
    ## add access-via []  e.g. layer[0].weight etc.
    def __getitem__(self, index): return self.layers[index]
    def __len__(self):  return len(self.layers)


###
# -- loss function layers
class MSELoss(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        return ((pred - target)*(pred - target)).sum(dim=0)
    
    
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
    
