from mintorch import Tensor 


x = Tensor( [1,2,3], requires_grad=True)
w = Tensor( [2,2,2], requires_grad=True)
b = Tensor( [3,3,3], requires_grad=True)

y = x*x + x*w - b 
y.backward()

print( x.grad ) 
print( w.grad ) 
print( b.grad )


####
### try with scalars only
x = Tensor( 1, requires_grad=True)
w = Tensor( 2, requires_grad=True)
b = Tensor( 3, requires_grad=True)

y = x*x + x*w - b 
y.backward()

print( x.grad ) 
print( w.grad ) 
print( b.grad )


###
## try matrix multiplcation

print( "---------" )
x = Tensor( [[1,2]], requires_grad=True )   # (1x2)
W = Tensor( [[.1,.2,.3],[.4,.5,.6]], requires_grad=True)   # (2x3)

y = x.mm(W)
y.backward()    # (1x3)

print( y ) 
print( x.grad ) 
print( W.grad )


print( "--" )
x = Tensor( [[1,2]], requires_grad=True )   # (1x2)
W = Tensor( [[.1,.2,.3],[.4,.5,.6]], requires_grad=True)   # (2x3)

y = x @ W
y.backward()    # (1x3)

print( y ) 
print( x.grad ) 
print( W.grad )


print( "bye" )

