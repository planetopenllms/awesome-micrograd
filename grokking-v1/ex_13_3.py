
from mintorch import Tensor 

x = Tensor([1,2,3,4,5], requires_grad=True)
y = Tensor([2,2,2,2,2], requires_grad=True)

z = x + y
z.backward()

print(x.grad)
print(y.grad)
print(z._creators)
print(z._op)

print( "-------------" )

a = Tensor([1,2,3,4,5]  , requires_grad=True)
b = Tensor([2,2,2,2,2])
c = Tensor([5,4,3,2,1])
d = Tensor([-1,-2,-3,-4,-5])

e = a + b
f = c + d
g = e + f

g.backward()

print(a.grad)



print("bye")