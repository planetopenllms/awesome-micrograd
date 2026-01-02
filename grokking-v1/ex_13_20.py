from mintorch import Tensor, nn, optim


data   = Tensor([1,2,1,2])   # data indices
target = Tensor([0,1,0,1])   # target indices

model = nn.Sequential(nn.Embedding(3,3), nn.Tanh(), nn.Linear(3,4))
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(parameters=model.parameters(), lr=0.1)

for i in range(10):    
    # Predict
    pred = model.forward(data)
    # Compare
    loss = criterion.forward(pred, target)
    # Learn
    loss.backward()
    optimizer.step()
    print("loss:", loss)

print("bye")

"""
loss: tensor(1.3551631, shape=(), ndim=0, dtype=float32)
loss: tensor(1.1603799, shape=(), ndim=0, dtype=float32)
loss: tensor(0.9230584, shape=(), ndim=0, dtype=float32)
loss: tensor(0.76862323, shape=(), ndim=0, dtype=float32)
loss: tensor(0.6851411, shape=(), ndim=0, dtype=float32)
loss: tensor(0.60728145, shape=(), ndim=0, dtype=float32)
loss: tensor(0.48626864, shape=(), ndim=0, dtype=float32)
loss: tensor(0.32166302, shape=(), ndim=0, dtype=float32)
loss: tensor(0.16975246, shape=(), ndim=0, dtype=float32)
loss: tensor(0.075536296, shape=(), ndim=0, dtype=float32)
"""