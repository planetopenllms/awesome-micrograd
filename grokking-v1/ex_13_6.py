
from mintorch import Tensor 

a = Tensor([1,2,3,4,5])
b = Tensor([2,2,2,2,2], requires_grad=True)
c = Tensor([5,4,3,2,1])

d = a + b
e = b + c
f = d + e
f.backward()

print( b.grad )  ## expected [2,2,2,2,2] NOT [1,1,1,1,1]


print("bye")