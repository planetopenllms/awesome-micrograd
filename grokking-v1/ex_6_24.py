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


model = nn.Sequential( nn.Linear(3, hidden_size, bias=False), nn.ReLU(),
                       nn.Linear(hidden_size, 1, bias=False) )

## note - use same (random) weight init as in orginal sample!!!
##    and random seed
np.random.seed(1)
model[0].weight.data = 2*np.random.random((3,hidden_size)) - 1
model[2].weight.data = 2*np.random.random((hidden_size,1)) - 1



criterion = nn.MSELoss()
optimizer = optim.SGD(parameters=model.parameters(), lr=lr)


for iteration in range(60):
   error = 0
   for i in range(len(streetlights)): 
      ## predict
      x =  Tensor(streetlights[i:i+1])
      y_hat = model( x ) 
      # Compare
      y =  Tensor(walk_vs_stop[i:i+1])
      loss = criterion(y_hat, y)
      error += loss.data
      # Learn
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
   if(iteration % 10 == 9):
      print( f"Error: {error}")


print( "bye" )

"""
Error: 0.49747171998023987
Error: 0.002649485832080245
Error: 1.4110898973740404e-06
Error: 6.523112605627546e-10
Error: 3.00026957488278e-13
Error: 1.5355111523647838e-14

--or--

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