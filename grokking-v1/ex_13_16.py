from mintorch import Tensor, SGD
from mintorch import nn


data   = Tensor([[0,0],[0,1],[1,0],[1,1]])
target = Tensor([[0],[1],[0],[1]])

model = nn.Sequential([nn.Linear(2,3), nn.Tanh(), nn.Linear(3,1), nn.Sigmoid()])
criterion = nn.MSELoss()

optim = SGD(parameters=model.get_parameters(), lr=1)

for i in range(10):    
    # Predict
    pred = model.forward(data)
    # Compare
    loss = criterion.forward(pred, target)
    # Learn
    loss.backward()
    optim.step()
    print("loss=", loss)


print( "bye" )