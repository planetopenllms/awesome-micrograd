from mintorch import Tensor 


data   = Tensor([[0,0],[0,1],[1,0],[1,1]])
target = Tensor([[0],[1],[0],[1]])

w = list()
w.append(Tensor.rand(2,3, requires_grad=True))
w.append(Tensor.rand(3,1, requires_grad=True))


for i in range(10):
    # Predict
    pred = data.mm(w[0]).mm(w[1])
    # Compare
    loss = ((pred - target)*(pred - target)).sum(dim=0)
    # Learn
    loss.backward()
    for w_ in w:
        w_.data -= w_.grad * 0.1
        w_.grad = None              ## set to zero (note: uses None like pytorch instead)

    print("------\nloss=", loss)



print("bye")