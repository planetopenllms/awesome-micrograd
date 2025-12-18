import numpy as np

class Tensor (object):    
    def __init__(self, data, requires_grad=False, _creators=(), _op=None):
        self.data = np.array(data, dtype=np.float32)   ### always - autoconvert to dtype float32!!!
        self.grad = None
        self.requires_grad = requires_grad
        ## note - minigrad by a. karpathy "filters out" duplicates using a set
        ##   to stay closer to book source - allow duplicates; get filtered out in toposort later
        ## self._prev = set(_creators) 
        self._creators = _creators
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc


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
      
        self.grad =  grad ## 1

        # go one variable at a time and apply the chain rule to get its gradient
        for v in reversed(topo):
            v._backward()

    
    def _input_grad( self, grad ):
        ## todo:
        ##   assert shape - self.data same as grad!!
        ##    input and input_grad MUST always be same shape
        ##    assert_shape here
        if(self.grad == None):
           self.grad = grad
        else:    ## accumulate gradients
           self.grad += grad          
       

    def _backward(self):
        print( f"--backward _op={self._op}, output_gradient={self.grad}")
        # grads must not have grads of their own
        ## assert self.grad.requires_grad == False
        
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
            ## note - hack - use Tensor() wrapper to get tensor copy WITHOUT require_grad=False 
            #                   might be possibly True in original!!!
            self._creators[0]._input_grad( output_grad * Tensor(self._creators[1].data) ) 
            self._creators[1]._input_grad( output_grad * Tensor(self._creators[0].data) )
        if(self._op == "mm"):
            c0 = self._creators[0]
            c1 = self._creators[1]
            new = output_grad.mm(c1.transpose())
            c0._input_grad( new )
            new = output_grad.transpose().mm(c0).transpose()
            c1._input_grad( new )
        if(self._op == "transpose"):
            self.creators[0]._input_grad( output_grad.transpose() )
        if(self._op == "neg"):
            self._creators[0]._input_grad( -output_grad ) ## or output_grad.__neg__()


    def __add__(self, other):
        ###  todo - double check - return to AND - why? why not?
        ##               see __mul__ for trouble in sample!!
        ## note - book uses and (NOT or) - why?
        if(self.requires_grad or other.requires_grad):
           return Tensor(self.data + other.data,
                       requires_grad=True,   
                      _creators=[self,other], _op="add")
        return Tensor(self.data + other.data)

    def __neg__(self):
        if(self.requires_grad):
            return Tensor(self.data * -1,
                          requires_grad=True,
                          _creators=[self], _op="neg")
        return Tensor(self.data * -1)

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

    def transpose(self):
        if(self.requires_grad):
            return Tensor(self.data.transpose(),
                          requires_grad=True,
                          _creators=[self], _op="transpose")
        return Tensor(self.data.transpose())
    
    def mm(self, x):
        ## double check requires_grad check again here!!!
        if(self.requires_grad or x.requires_grad):
            return Tensor(self.data.dot(x.data),   ## change .dot to mm() too if possible???
                          requires_grad=True,
                          _creators=[self,x], _op="mm")
        return Tensor(self.data.dot(x.data))


    def __repr__(self):
        return str(self.data.__repr__())
    
    def __str__(self):
        return  f"tensor({self.data.__str__()}, shape={self.data.shape}, ndim={self.data.ndim}, dtype={self.data.dtype})"
    