from mintorch import Tensor, nn  

# Quick demo
d = nn.Dropout(p=0.5)
x = Tensor.rand( 3, 4 )

print("training mode (default):", d.training)
out_train = d(x)
print("input:\n", x)
print("output (dropout applied):\n", out_train)

d.eval()
print("\nafter eval(), training:", d.training)
out_eval = d(x)
print("output in eval (should equal input):\n", out_eval)


### try grad

x = Tensor.rand( 3, 4, requires_grad=True)
y = x.dropout(p=0.5)

print( "x", x )
print( "y", y )
y.backward()
print( "x.grad", x.grad )


print("bye")
