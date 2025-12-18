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


    def backward(self, grad):
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

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad =  grad ## 1
        for v in reversed(topo):
            v._backward()

    
    def _input_grad( self, grad ):
        if(self.grad == None):
           self.grad = grad
        else:    ## accumulate gradients
           self.grad += grad          


    def _backward(self):
        print( f"--backward _op={self._op}, output_gradient={self.grad}")
        output_grad = self.grad
        if(self._op == "add"):
            self._creators[0]._input_grad( output_grad )
            self._creators[1]._input_grad( output_grad )


    def __add__(self, other):
        ## note - book uses and (NOT or) - why?
        if(self.requires_grad or other.requires_grad):
           return Tensor(self.data + other.data,
                       requires_grad=True,   
                      _creators=[self,other], _op="add")
        return Tensor(self.data + other.data)


    def __repr__(self):
        return str(self.data.__repr__())
    
    def __str__(self):
        return  f"tensor({self.data.__str__()}, shape={self.data.shape}, ndim={self.data.ndim}, dtype={self.data.dtype})"
    