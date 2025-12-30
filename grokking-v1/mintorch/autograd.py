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



def ensure_tensor(x):
    if isinstance(x, Tensor):
        return x
    
    ## todo - add (auto-)conversion from
    #           torch.tensor too; use  .numpy() or such - why? why not? 
    
    return Tensor( x )



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
    def randn(cls, *size, requires_grad=False):
        """random normal dist with mean 0 and std.dev 1 helper"""
        return cls( np.random.randn( *size ), requires_grad=requires_grad)

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
        if(self._op == "relu"):
            self._creators[0]._input_grad(output_grad * (self.data > 0))
        if(self._op == "dropout"):
            self._creators[0]._input_grad(output_grad * self.mask)
        if(self._op == "cross_entropy"):
            ## todo/check - why no output_grad in formula?
            ##      assume always loss? that is, start of backward calc?
            dx = self.softmax_output - self.target_dist
            self._creators[0]._input_grad( dx )


    def __add__(self, other):
        ## fixes
        ##  AttributeError: 'int' object has no attribute 'requires_grad'
        other = ensure_tensor(other)
        if(self.requires_grad or other.requires_grad):
           return Tensor(self.data + other.data,
                       requires_grad=True,   
                      _creators=[self,other], _op="add")
        return Tensor(self.data + other.data)
    def __radd__(self, other):
        ##  fixes
        ## unsupported operand type(s) for +: 'int' and 'Tensor'
        ##  note - radd is called with self being the right-hand operand
        return ensure_tensor(other).__add__(self)

    def __sub__(self, other):
        other = ensure_tensor(other)
        if(self.requires_grad or other.requires_grad):
            return Tensor(self.data - other.data,
                          requires_grad=True,
                          _creators=[self,other], _op="sub")
        return Tensor(self.data - other.data)
    def __rsub__(self, other):
        return ensure_tensor(other).__sub__(self)

    def __neg__(self):
        if(self.requires_grad):
            return Tensor(-self.data,   ## or use *-1
                          requires_grad=True,
                          _creators=[self], _op="neg")
        return Tensor(-self.data)

    ###
    #  todo - add ensure_tensor check to ops
    #          and add reverse operator too!!! 
    def __mul__(self, other):
        if(self.requires_grad or other.requires_grad):
            return Tensor(self.data * other.data,
                          requires_grad=True,
                          _creators=[self,other], _op="mul")
        return Tensor(self.data * other.data)   

    def sum(self, dim):
        new_data = self.data.sum(axis=dim)
        if(self.requires_grad):
            return Tensor(new_data,
                          requires_grad=True,
                          _creators=[self], _op="sum_"+str(dim))
        return Tensor(new_data)
    
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

    ## todo - check mm vs matmul in pytorch
    ##      one works with broadcast? 
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

    def relu(self):
        if(self.requires_grad):
            return Tensor( np.maximum(0, self.data),
                            requires_grad=True,
                            _creators=[self],
                            _op="relu")
        return Tensor(np.maximum(0, self.data))
   

    def dropout(self, p=0.5):
        keep_prob = 1.0 - p
        # create mask of same shape with entries 1/keep_prob 
        # where kept, 0 where dropped
        mask = (np.random.rand(*self.data.shape) < keep_prob)
        mask = mask / keep_prob  # inverted dropout scaling
        new_data =  self.data * mask
        if(self.requires_grad):
           out = Tensor( new_data,
                           requires_grad=True,
                           _creators=[self],
                           _op="dropout")
           out.mask = mask
           return out
        return Tensor(new_data)


    def cross_entropy(self, target_indices):
        temp = np.exp(self.data)  # elementwise
        softmax_output = temp / np.sum(temp,
                                       axis=-1,
                                       keepdims=True)
        ## print("softmax_output:", softmax_output.shape, softmax_output )
        ## note: indices must be integer!!!!
        t = target_indices.data.astype(int) 
        ## print( "t:", t.shape, t )
        ## make one-hot encoded targets
        ## print( "np.eye():", np.eye(self.data.shape[-1]) )
        # target_dist = np.eye(p.shape[1])[t]
        target_dist = np.eye(self.data.shape[-1])[t]
        ## print( "target_dist:", target_dist.shape, target_dist )
        ##  note - uses mean() thus reduces dimension!!
        ##                     check dim of returned loss!!
        ##     check/todo -  also add mean() to  MSELoss (why? why not?)
        ## loss = -(np.log(p) * (target_dist)).sum(axis=1).mean()
        loss = -(np.log(softmax_output) * (target_dist)).sum(axis=-1).mean()
    
        if(self.requires_grad):
            out = Tensor(loss,
                         requires_grad=True,
                         _creators=[self],
                         _op="cross_entropy")
            out.softmax_output = softmax_output
            out.target_dist    = target_dist  # one-hot encoded (from int classes)
            return out
        return Tensor(loss)


    def __repr__(self):
        return str(self.data.__repr__())
    
    def __str__(self):
        return  f"tensor({self.data.__str__()}, shape={self.data.shape}, ndim={self.data.ndim}, dtype={self.data.dtype})"
    

