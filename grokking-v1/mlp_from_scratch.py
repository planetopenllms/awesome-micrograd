##########
#  mintorch version of 
#     multi-layer perceptron (mlp) from scratch with pytorch tensors
# 
from mintorch import Tensor, nn, optim
import numpy as np

# N is batch size, D_in is input dimension;
# H is hidden dimension, D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10


# create random input and output data
x = Tensor.randn( N, D_in )
y = Tensor.randn( N, D_out )

lr =   0.000001     ## 1e-6

layer1 = nn.Linear(D_in, H)
layer2 = nn.Linear(H, D_out)

## note - use same weight init as in orginal sample!!!
layer1.weight.data = np.random.randn(D_in, H)
layer2.weight.data = np.random.randn(H, D_out)


model = nn.Sequential( [layer1, nn.ReLU(),
                        layer2] )


criterion = nn.MSELoss()
optimizer = optim.SGD(parameters=model.parameters(), lr=lr)


for t in range(500):
    # forward pass: compute predicated y
    y_pred = model( x ) 

    # compute and print loss
    loss = criterion(y_pred, y)
    print(t, loss.data.sum())

    # backprop to compute gradients of w1 and w2 with respect to loss
    optimizer.zero_grad()
    loss.backward()
    # update weights using gradient descent
    optimizer.step()
    

print( "bye" )