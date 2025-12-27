###
#  multi-layer perceptron (mlp) from scratch with pytorch tensors
import torch


# N is batch size, D_in is input dimension;
# H is hidden dimension, D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# create random input and output data
x = torch.randn( N, D_in )
y = torch.randn( N, D_out )

# randomly initialize weights
w1 = torch.randn(D_in, H)
w2 = torch.randn(H, D_out)

learning_rate = 0.000001    # 1e-6
for t in range(500):
    # forward pass: compute predicated y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    print(t, loss)

    # backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.mm( grad_y_pred )

    grad_h_relu = grad_y_pred.mm( w2.T )
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0    ## inline relu compute
    grad_w1 = x.T.mm( grad_h )

    # update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2


print( "bye" )