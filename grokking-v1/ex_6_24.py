##########
#  mintorch version of ex_6_24.py

from mintorch import Tensor, nn, optim
import numpy as np

## data (batch x features)
streetlights = np.array( [[ 1, 0, 1 ],
                          [ 0, 1, 1 ],
                          [ 0, 0, 1 ],
                          [ 1, 1, 1 ] ] )
## targets (batch x target)
walk_vs_stop = np.array([[1], [1], [0], [0]])


lr = 0.1
hidden_size = 4

model = nn.Sequential( [nn.Linear(3, hidden_size), nn.ReLU(),
                        nn.Linear( hidden_size, 1)] )

criterion = nn.MSELoss()
optimizer = optim.SGD(parameters=model.parameters(), lr=lr)


for iteration in range(60):
   layer_2_error = 0
   for i in range(len(streetlights)): 
      ## predict
      x = Tensor(streetlights[i:i+1])
      y = Tensor(walk_vs_stop[i:i+1])
      y_hat = model( x ) 
      # Compare
      loss = criterion(y_hat, y)
      layer_2_error += loss.data   ## use loss.item() ???
      # Learn
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
   if(iteration % 10 == 9):
      print( f"Error: {layer_2_error}")


print( "bye" )