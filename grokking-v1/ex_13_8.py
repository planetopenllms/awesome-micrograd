from mintorch import Tensor 

a = Tensor([1,2,3,4,5], requires_grad=True)
b = Tensor([2,2,2,2,2], requires_grad=True)
c = Tensor([5,4,3,2,1], requires_grad=True)

d = a + (-b)
e = (-b) + c
f = d + e

f.backward()

print(b.grad )   ## expeced [-2,-2,-2,-2,-2]

print( "bye" )

