## test (auto-)conversion to tensor from numpy or torch.tensor etc.

from mintorch import Tensor

import numpy as np

## add
x = Tensor(1)

print( 1+x )   # __radd__(self,other)   
print( x+1 )   # __add__(self,other)

print( np.array(1)+x )   # __radd__(self,other)   
print( x+np.array(1) )   # __add__(self,other)


print( 1-x )   #  __rsub__(self,other)
print( x-1 )   #  __sub__(self,other)

print( np.array(1)-x )   #  __rsub__(self,other)
print( x-np.array(1) )   #  __sub__(self,other)


print("bye")

