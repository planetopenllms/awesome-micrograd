
class SGD():
    def __init__(self, parameters, lr=0.1):
        self.parameters = parameters
        self.lr = lr
    
    def zero_grad(self):
        for p in self.parameters:
            p.grad = None   ## todo/fix:  p.zero_grad() !!!
        
    def step(self):     
        for p in self.parameters:
            p.data -= p.grad * self.lr
            
