from mintorch import Tensor, nn, optim


data   = Tensor([[0,0],[0,1],[1,0],[1,1]])
target = Tensor([[0],[1],[0],[1]])

model = nn.Sequential([nn.Linear(2,3), nn.Linear(3,1)])

optimizer = optim.SGD(parameters=model.parameters(), lr=0.05)

for i in range(10):    
    # Predict
    pred = model(data)
    # Compare    
    loss = ((pred - target)*(pred - target)).sum(dim=0)    
    # Learn
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("loss=", loss)

print( "bye" )

