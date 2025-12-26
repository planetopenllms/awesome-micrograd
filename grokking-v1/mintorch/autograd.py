import numpy as np


## helper to expand np array
##      check if there's something builtin
#       from the book Grokking Deep Learning
def _np_expand_old(data, dim, copies):
    trans_cmd = list(range(0,len(data.shape)))
    trans_cmd.insert(dim,len(data.shape))
    new_data = data.repeat(copies).reshape(list(data.shape) + [copies]).transpose(trans_cmd)
    return new_data

##
## todo - replace with simpler version - why? why not?
##          check/assert if both are the same
def _np_expand(data, dim, copies):
    # bonus - allow negative dims: convert to insertion index in [0, data.ndim]
    # if dim < 0:
    #    dim += data.ndim + 1
    return np.repeat(np.expand_dims(data, axis=dim), copies, axis=dim)


class Tensor():   

    def __init__(self, data, requires_grad=False, _creators=(), _op=None):
        self.data = np.array(data, dtype=np.float32)   ### always - autoconvert to dtype float32!!!
        self.grad = None
        self.requires_grad = requires_grad
        ## note - minigrad by a. karpathy "filters out" duplicates using a set
        ##   to stay closer to book source - allow duplicates; get filtered out in toposort later
        ## self._prev = set(_creators) 
        self._creators = _creators
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc


    ### check - rename to uniform or add alias - why? why not?
    @classmethod
    def rand(cls, *size, requires_grad=False):
        """random uniform dist [0,1] helper"""
        return cls( np.random.rand( *size ), requires_grad=requires_grad)

    @classmethod
    def zeros(cls, *size, requires_grad=False):
        return cls( np.zeros( *size ), requires_grad=requires_grad)
        


    def backward(self, grad=None):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._creators:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
   
        if grad is None:
            grad = np.ones_like( self.data )

        ## note - grad is NOT wrapped / a tensor!
        ##           MUST be raw numpy ndarray!!
        assert isinstance( grad, (np.ndarray,np.generic)), f"grad type is {type(grad)}"
        self.grad =  grad    ## 1

        # go one variable at a time and apply the chain rule to get its gradient
        for v in reversed(topo):
            v._backward()

    
    def _input_grad( self, grad ):
        ## note - grad is NOT wrapped / a tensor!
        ##           MUST be raw numpy ndarray (or numpy scalar)!!
        assert isinstance( grad, (np.ndarray,np.generic)), f"grad type is {type(grad)}"
 
        ## todo:
        ##   assert shape - self.data same as grad!!
        ##    input and input_grad MUST always be same shape
        ##    assert_shape here
        if self.grad is None:
           self.grad = grad
        else:    ## accumulate gradients
           self.grad += grad          
       

    def _backward(self):
        ## print( f"--backward _op={self._op}, output_gradient={self.grad}")
        
        ## todo
        ##  assert shape - self.data same as self.grad!!
        ##    output and output_grad MUST always be same shape
        ##  assert_shape here
 
        output_grad = self.grad
        
        if(self._op == "add"):
            self._creators[0]._input_grad( output_grad )
            self._creators[1]._input_grad( output_grad )
        if(self._op == "sub"):
            self._creators[0]._input_grad( output_grad )
            self._creators[1]._input_grad( -output_grad )
        if(self._op == "mul"):
            self._creators[0]._input_grad( output_grad * self._creators[1].data ) 
            self._creators[1]._input_grad( output_grad * self._creators[0].data )
        if(self._op == "mm"):
            c0 = self._creators[0]
            c1 = self._creators[1]
            c0._input_grad( np.matmul(output_grad, c1.data.T) )
            c1._input_grad( np.matmul(output_grad.T, c0.data).T )
        if(self._op == "transpose"):
            self._creators[0]._input_grad( output_grad.T )
        if(isinstance(self._op,str) and self._op.startswith( "sum_")):
            dim = int(self._op.split("_")[1])
            self._creators[0]._input_grad( 
                    _np_expand( output_grad, dim, self._creators[0].data.shape[dim]))
        if(isinstance(self._op, str) and self._op.startswith("expand_")):
            dim = int(self._op.split("_")[1])
            self._creators[0]._input_grad( output_grad.sum(axis=dim) )
        if(self._op == "neg"):
            self._creators[0]._input_grad( -output_grad )
        if(self._op == "sigmoid"):
            ones = np.ones_like(output_grad)
            self._creators[0]._input_grad(output_grad * (self.data * (ones - self.data)))
        if(self._op == "tanh"):
            ones = np.ones_like(output_grad)
            self._creators[0]._input_grad(output_grad * (ones - (self.data * self.data)))



    def __add__(self, other):
        if(self.requires_grad or other.requires_grad):
           return Tensor(self.data + other.data,
                       requires_grad=True,   
                      _creators=[self,other], _op="add")
        return Tensor(self.data + other.data)

    def __neg__(self):
        if(self.requires_grad):
            return Tensor(-self.data,   ## or use *-1
                          requires_grad=True,
                          _creators=[self], _op="neg")
        return Tensor(-self.data)

    def __sub__(self, other):
        if(self.requires_grad or other.requires_grad):
            return Tensor(self.data - other.data,
                          requires_grad=True,
                          _creators=[self,other], _op="sub")
        return Tensor(self.data - other.data)
    
    def __mul__(self, other):
        if(self.requires_grad or other.requires_grad):
            return Tensor(self.data * other.data,
                          requires_grad=True,
                          _creators=[self,other], _op="mul")
        return Tensor(self.data * other.data)   

    def sum(self, dim):
        if(self.requires_grad):
            return Tensor(self.data.sum(axis=dim),
                          requires_grad=True,
                          _creators=[self], _op="sum_"+str(dim))
        return Tensor(self.data.sum(axis=dim))
    
    def expand(self, dim, copies):
        new_data = _np_expand( self.data, dim, copies )     
        if(self.requires_grad):
            return Tensor(new_data,
                          requires_grad=True,
                          _creators=[self], _op="expand_"+str(dim))
        return Tensor(new_data)

    def transpose(self):
        if(self.requires_grad):
            return Tensor(self.data.T,
                          requires_grad=True,
                          _creators=[self], _op="transpose")
        return Tensor(self.data.T)
   
    @property
    def T(self):
        return self.transpose()
   

    def __matmul__(self,x):
        if(self.requires_grad or x.requires_grad):
            return Tensor(np.matmul(self.data, x.data),    
                          requires_grad=True,
                          _creators=[self,x], _op="mm")
        return Tensor(np.matmul(self.data, x.data))

    def mm(self, x):
        return self.__matmul__(x)


    def sigmoid(self):
        if(self.requires_grad):
            return Tensor(1 / (1 + np.exp(-self.data)),
                          requires_grad=True,
                          _creators=[self],
                          _op="sigmoid")
        return Tensor(1 / (1 + np.exp(-self.data)))

    def tanh(self):
        if(self.requires_grad):
            return Tensor(np.tanh(self.data),
                          requires_grad=True,
                          _creators=[self],
                          _op="tanh")
        return Tensor(np.tanh(self.data))


    def __repr__(self):
        return str(self.data.__repr__())
    
    def __str__(self):
        return  f"tensor({self.data.__str__()}, shape={self.data.shape}, ndim={self.data.ndim}, dtype={self.data.dtype})"
    


class SGD():
    def __init__(self, parameters, lr=0.1):
        self.parameters = parameters
        self.lr = lr
    
    def zero(self):
        for p in self.parameters:
            p.grad = None
        
    def step(self, zero=True):     
        for p in self.parameters:
            p.data -= p.grad * self.lr
            if(zero):
                p.grad = None

