from mintorch import Tensor, optim  


data   = Tensor([[0,0],[0,1],[1,0],[1,1]])
target = Tensor([[0],[1],[0],[1]])

w = list()
w.append(Tensor.rand(2,3, requires_grad=True))
w.append(Tensor.rand(3,1, requires_grad=True))


optimizer = optim.SGD(parameters=w, lr=0.1)

for i in range(10):
    # Predict
    pred = data.mm(w[0]).mm(w[1])
    # Compare
    loss = ((pred - target)*(pred - target)).sum(dim=0)   
    # Learn
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("loss=", loss)


print("bye")