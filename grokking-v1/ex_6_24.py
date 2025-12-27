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


lr = 0.2
hidden_size = 4


layer1 = nn.Linear(3, hidden_size)
layer2 = nn.Linear(hidden_size, 1)

## note - use same weight init as in orginal sample!!!
layer1.weight.data = 2*np.random.random((3,hidden_size)) - 1
layer2.weight.data = 2*np.random.random((hidden_size,1)) - 1


model = nn.Sequential( [layer1, nn.ReLU(),
                        layer2] )

criterion = nn.MSELoss()
optimizer = optim.SGD(parameters=model.parameters(), lr=lr)


for iteration in range(60):
   layer_2_error = 0
   for i in range(len(streetlights)): 
      ## predict
      x = Tensor(streetlights[i:i+1])
      y_hat = model( x ) 
      # Compare
      y = Tensor(walk_vs_stop[i:i+1])
      loss = criterion(y_hat, y)
      layer_2_error += loss.data.sum()    ## use loss.item() ???
      # Learn
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
   if(iteration % 10 == 9):
      print( f"Error: {layer_2_error}")


print( "bye" )

"""
Error: 1.3428703546524048
Error: 1.1580779552459717
Error: 0.9731759428977966
Error: 0.8535886406898499
Error: 0.8214600682258606
Error: 0.8197005391120911

--or--

Error: 0.48454803228378296
Error: 0.04550359770655632
Error: 0.004576224833726883
Error: 0.0003909869119524956
Error: 2.9069948141113855e-05
Error: 2.4260564259748207e-06
"""