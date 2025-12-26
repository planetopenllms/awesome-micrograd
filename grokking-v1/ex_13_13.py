from mintorch import Tensor, SGD 
from mintorch import nn



data   = Tensor([[0,0],[0,1],[1,0],[1,1]])
target = Tensor([[0],[1],[0],[1]])

model = nn.Sequential([nn.Linear(2,3), nn.Linear(3,1)])

optim = SGD(parameters=model.get_parameters(), lr=0.05)

for i in range(10):    
    # Predict
    pred = model.forward(data)
    # Compare
    loss = ((pred - target)*(pred - target)).sum(dim=0)    
    # Learn
    loss.backward()
    optim.step()
    print("loss=", loss)

print( "bye" )

