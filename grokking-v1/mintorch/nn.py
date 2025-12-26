from .autograd import Tensor
import numpy as np


class Layer():
    def __init__(self):
        self.parameters = list()  ## note: same as []
        
    def get_parameters(self):
        return self.parameters


class Linear(Layer):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        ## uses he/()-init with normal dist
        ##  fix use Tensor.normal( 0, var? or std=  sqrt(2.0/n_inputs))
        W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0/(n_inputs))
        self.weight = Tensor(W, requires_grad=True)
        self.bias   = Tensor.zeros(n_outputs, requires_grad=True)
        
        self.parameters.append(self.weight)
        self.parameters.append(self.bias)

    def forward(self, input):
        ## note: no "broadcast" used for bias; bias gets expanded for batch dim
        return input.mm(self.weight)+self.bias.expand(dim=0,copies=len(input.data))
    
class Sequential(Layer):    
    def __init__(self, layers=list()):
        super().__init__()
        self.layers = layers
    
    def add(self, layer):
        self.layers.append(layer)
        
    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def get_parameters(self):
        params = list()
        for l in self.layers:
            params += l.get_parameters()
        return params
    

###
# -- loss function layers
class MSELoss(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        return ((pred - target)*(pred - target)).sum(dim=0)
    
    
###
# -- activation function layers
class Tanh(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return input.tanh()
    
class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return input.sigmoid()