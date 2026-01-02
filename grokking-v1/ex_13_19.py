from mintorch import Tensor, nn, optim


data   = Tensor([1,2,1,2])
target = Tensor([[0],[1],[0],[1]])


embed = nn.Embedding(5,3)   ## vocab_size x embed_dim
model = nn.Sequential(embed, nn.Tanh(), nn.Linear(3,1), nn.Sigmoid())
criterion = nn.MSELoss()

optimizer = optim.SGD(parameters=model.parameters(), lr=0.5)

for i in range(10):    
    # Predict
    pred = model.forward(data)
    # Compare
    loss = criterion.forward(pred, target)
    # Learn
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("loss=", loss)


## print embedding weight matrix
print( "embed.weight:", embed.weight )


print("bye")

"""
loss= tensor(0.25420788, shape=(), ndim=0, dtype=float32)
loss= tensor(0.2092765, shape=(), ndim=0, dtype=float32)
loss= tensor(0.16865471, shape=(), ndim=0, dtype=float32)
loss= tensor(0.12734725, shape=(), ndim=0, dtype=float32)
loss= tensor(0.090577915, shape=(), ndim=0, dtype=float32)
loss= tensor(0.06346171, shape=(), ndim=0, dtype=float32)
loss= tensor(0.045674346, shape=(), ndim=0, dtype=float32)
loss= tensor(0.03429759, shape=(), ndim=0, dtype=float32)
loss= tensor(0.026830545, shape=(), ndim=0, dtype=float32)
loss= tensor(0.021722171, shape=(), ndim=0, dtype=float32)
"""