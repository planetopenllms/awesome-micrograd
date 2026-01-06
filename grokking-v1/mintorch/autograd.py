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


#########
###  batch version for softmax  
###     (batch x classes)  or
###   xxx  (batch x n_tokens x classes)  -- keep this too? why? why not?
##    xxx        was axis=1 (now uses axis=last, that is axis=-1)                          
def _np_softmax(Z):
    """
    Z: (batch x classes) logits
    returns: softmax probabilities
    """
    # shift for numerical stability  (max value is zero, all others shifted to negative!)
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    ## print( "Z_shifted:", Z_shifted )

    exp_Z = np.exp(Z_shifted)
    S = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    return S

def _np_onehot(indices, classes):
    """
    encode labels (indices) e.g. [0,3,1] to one-hot encoded vectors
    note - adds a dimension e.g.
    (batch) => (batch x classes)
    """
    ## note: indices must be integer!!!!
    indices = indices.astype(int) 
    ## make one-hot encoded targets
    ## print( "np.eye():", np.eye(self.data.shape[-1]) )
    # target_dist = np.eye(p.shape[1])[t]
    target = np.eye( classes )[ indices ]
    return target

###
###  batch version for combinded softmax and cross_entroy
def _np_softmax_cross_entropy(Z, indices):
    """
    Z: (batch x classes) logits
    Y: indices 
    returns: scalar loss, softmax probabilities
    """
    S = _np_softmax( Z )
 
    batch_size   = Z.shape[0]
 
    # note: add small epsilon (1e-15) for safety
    logprobs = -np.log(S[np.arange(batch_size), indices] + 1e-15)

    ## same as np.mean(logprobs)
    loss = np.sum(logprobs) / batch_size

    return loss, S




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
        
        if(self._op == "tanh"):
            ones = np.ones_like(output_grad)
            self._creators[0]._input_grad(output_grad * (ones - (self.data * self.data)))
        if(self._op == "relu"):
            self._creators[0]._input_grad(output_grad * (self.data > 0))
        if(self._op == "dropout"):
            self._creators[0]._input_grad(output_grad * self.mask)
        if(self._op == "mse"):   # mean squared error (mse)
            ## todo/check - why no output_grad in formula?
            ##      assume always loss? that is, start of backward calc?
            #
            #    note: PyTorch backprop is
            ##  layer_2_delta = 2 * (y_hat - y) / (batch_size * output_dim)
            ##   missing division by batch size!!!  (and bonus output_dim!!!)
            ##
            dx = 2*(self._creators[0].data-self._creators[1].data)   ## 2*(y_hat-y)
            self._creators[0]._input_grad( dx ) ## dl/dy_hat
        if(self._op == "cross_entropy"):
            ## todo/check - why no output_grad in formula?
            ##      assume always loss? that is, start of backward calc?
            ##
            ##  note:
            ##  Important difference from MSE
            ##  There is no extra division by C (output_dim).
            ##   Cross-entropy already sums over classes inside each sample.
            ##
            ##  note: PyTorch backprop is
            ##    p = softmax(logits)
            ##    grad_logits = (p - one_hot(targets)) / B
            ##
            ##   missing division by B (batch) ?? why? why not?
            S = self.softmax_output.copy()   ## note: will change softmax_output inplace later; make copy
            batch_size = S.shape[0]
            target_indices  = self._creators[1].data.astype(int)
            S[np.arange(batch_size), target_indices] -= 1
            self._creators[0]._input_grad( S )   # dL/dy_hat
        if(self._op == "index_select"):
            new_grad = np.zeros_like(self._creators[0].data)
            ## note: indices must be integer!!!!
            indices_ = self._creators[1].data.flatten().astype(int)
            grad_ = output_grad.reshape(len(indices_), -1)
            for i in range(len(indices_)):
                new_grad[indices_[i]] += grad_[i]
            self._creators[0]._input_grad( new_grad )


    def __add__(self, other):
        ## fixes
        ##  AttributeError: 'int' object has no attribute 'requires_grad'
        other = ensure_tensor(other)
        if(self.requires_grad or other.requires_grad):        
           out = Tensor(self.data + other.data,
                        requires_grad=True,   
                        _creators=[self,other], _op="add")
           def _backward():
               self._input_grad( out.grad )
               other._input_grad( out.grad )  
           out._backward = _backward
           return out
        return Tensor(self.data + other.data)
   
    def __radd__(self, other):
        ##  fixes
        ## unsupported operand type(s) for +: 'int' and 'Tensor'
        ##  note - radd is called with self being the right-hand operand
        return ensure_tensor(other).__add__(self)

    def __sub__(self, other):
        other = ensure_tensor(other)
        if(self.requires_grad or other.requires_grad):
            out = Tensor(self.data - other.data,
                          requires_grad=True,
                          _creators=[self,other], _op="sub")
            def _backward():
                self._input_grad( out.grad )
                other._input_grad( -out.grad )
            out._backward = _backward
            return out
        return Tensor(self.data - other.data)
    def __rsub__(self, other):
        return ensure_tensor(other).__sub__(self)

    def __neg__(self):
        if(self.requires_grad):
            out = Tensor(-self.data,   ## or use *-1 - why? why not?
                          requires_grad=True,
                          _creators=[self], _op="neg")
            def _backward():
                self._input_grad( -out.grad ) 
            out._backward = _backward
            return out
        return Tensor(-self.data)

    ###
    #  todo - add ensure_tensor check to ops
    #          and add reverse operator too!!! 
    def __mul__(self, other):
        if(self.requires_grad or other.requires_grad):
            out = Tensor(self.data * other.data,
                          requires_grad=True,
                          _creators=[self,other], _op="mul")
            def _backward():
                self._input_grad( out.grad * other.data ) 
                other._input_grad( out.grad * self.data )
            out._backward = _backward
            return out
        return Tensor(self.data * other.data)   

    def sum(self, dim):
        new_data = self.data.sum(axis=dim)
        if(self.requires_grad):
            out = Tensor(new_data,
                          requires_grad=True,
                          _creators=[self], _op=f"sum_{dim}")
            def _backward():
                self._input_grad( 
                    _np_expand( out.grad, dim=dim, copies=self.data.shape[dim]))
            out._backward = _backward
            return out
        return Tensor(new_data)
    
    def expand(self, dim, copies):
        new_data = _np_expand( self.data, dim, copies )     
        if(self.requires_grad):
            out = Tensor(new_data,
                          requires_grad=True,
                          _creators=[self], _op=f"expand_{dim}")
            def _backward():
                self._input_grad( out.grad.sum(axis=dim) )
            out._backward = _backward
            return out
        return Tensor(new_data)


    def transpose(self):
        if(self.requires_grad):
            out = Tensor(self.data.T,
                          requires_grad=True,
                          _creators=[self], _op="transpose")
            def _backward():
               self._input_grad( out.grad.T )
            out._backward = _backward
            return out
        return Tensor(self.data.T)
   
    @property
    def T(self):
        return self.transpose()
   

    def __matmul__(self,x):
        if(self.requires_grad or x.requires_grad):
            out = Tensor(np.matmul(self.data, x.data),    
                          requires_grad=True,
                          _creators=[self,x], _op="mm")
            def _backward():
                self._input_grad( np.matmul(out.grad, x.data.T) )
                x._input_grad( np.matmul(out.grad.T, self.data).T )
            out._backward = _backward
            return out
        return Tensor(np.matmul(self.data, x.data))

    ## todo - check mm vs matmul in pytorch
    ##      one works with broadcast? 
    def mm(self, x):
        return self.__matmul__(x)


    def sigmoid(self):
        if(self.requires_grad):
            out = Tensor(1 / (1 + np.exp(-self.data)),
                          requires_grad=True,
                          _creators=[self],
                          _op="sigmoid")
            def _backward():
                ones = np.ones_like(out.grad)
                self._input_grad(out.grad * (self.data * (ones - self.data)))
            out._backward = _backward
            return out
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

    def mse(self, target):
        ## mean squared error (higher-level function)
        ##    for batches
        batch_size   = self.data.shape[0] 
        loss = np.sum(np.power(self.data-target.data, 2)) / batch_size
        if(self.requires_grad):
           return Tensor(loss,
                         requires_grad=True,
                         _creators=[self,target],
                         _op="mse")
        return Tensor(loss)
    
    def cross_entropy(self, target_indices):
        ## note: target_indices must be integer!!!!
        loss, softmax_output = _np_softmax_cross_entropy( self.data, 
                                                          target_indices.data.astype(int)
                                                        )      
        if(self.requires_grad):
            out = Tensor(loss,
                         requires_grad=True,
                         _creators=[self, target_indices],
                         _op="cross_entropy")
            out.softmax_output = softmax_output
            return out
        return Tensor(loss)

    def index_select(self, indices):
        ## note: indices must be integer!!!!
        new_data = self.data[indices.data.astype(int)]
        if(self.requires_grad):
            return Tensor(new_data,
                           requires_grad=True,
                           _creators=[self, indices],
                           _op="index_select")
        return Tensor(new_data)


    def __repr__(self):
        return str(self.data.__repr__())
    
    def __str__(self):
        return  f"tensor({self.data.__str__()}, shape={self.data.shape}, ndim={self.data.ndim}, dtype={self.data.dtype})"
    

