# Awesome Perceptrons

An awesome collection of perceptrons - single-layer (OR/AND) and multi-layer (XOR)



## Single-Layer Perceptron (2 Inputs, 1 Output)

Train on OR/AND (linear)

Logical OR Problem:

| Input 1 | Input 2 | Output (OR) |
|---------|---------|-------------|
|    0    |    0    |      0      |
|    0    |    1    |      1      |
|    1    |    0    |      1      |
|    1    |    1    |      1      |

Logical AND Problem:

| Input 1 | Input 2 | Output (AND) |
|---------|---------|-------------|
|    0    |    0    |      0      |
|    0    |    1    |      0      |
|    1    |    0    |      0      |
|    1    |    1    |      1      |


for X (inputs) use:

```python
X_train = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

```

for y (outputs/labels/targets) use:

```python
Y_or   = [0, 1, 1, 1]  # Output labels for OR operation (linear)
Y_and  = [0, 0, 0, 1]  # Output labels for AND operation (linear)
# Y_xor =  = [0, 1, 1, 0]  # Output labels for XOR operation (non-linear)
```



Step functions - Heaviside

```python
def heaviside(x):
    return 1 if x > 0 else 0
```

note - for step function the error is 1 / -1 or 0 for bingo! no error.



### Vanilla

```python
class Perceptron:
    # initialize w(eights) and b(ias)
    def __init__(self):
        self.w = [0,0]
        self.b =  0
        self.step_function = heaviside

    def predict(self, x):
        total = self.w[0] * x[0] +\
                self.w[1] * x[1] + self.b
        return self.step_function(total)

    # Train the perceptron using the perceptron learning rule
    def train(self, X, Y, epochs=10, learning_rate=0.1):
        for epoch in range(epochs):
            for x, y in zip(X, Y):
                y_hat = self.predict(x)
                error = y - y_hat
                # update weights and bias based on the error
                self.w[0] += learning_rate * error * x[0]
                self.w[1] += learning_rate * error * x[1]
                self.b    += learning_rate * error
```



### With [NumPy](https://numpy.org)


```python
import numpy as np

class Perceptron:
    def __init__(self):
        self.w = np.zeros(2)
        self.b = 0
        self.step_function = heaviside

    def predict(self, x):
        total = np.dot(x, self.w) + self.b
        return self.step_function(total)

    def train(self, X, Y, epochs=10, learning_rate=0.1):
        for epoch in range(epochs):
            for x, y in zip(X, Y):
                y_hat = self.predict(x)
                error = y - y_hat
                # Update weights and bias based on the error
                self.w += learning_rate * error * x
                self.b += learning_rate * error
```




for training use:

```python
# Define the training data for logical ops
X_train = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

Y_or   = [0, 1, 1, 1]  # Output labels for OR operation
# Y_and  = [0, 0, 0, 1]  # Output labels for AND operation
# Y_xor  = [0, 1, 1, 0]  # Output labels for XOR operation

perceptron = Perceptron()
perceptron.train(X_train, Y_or)

# Test the trained perceptron
print("Testing the trained perceptron:")
for x in X_train:
    y_hat = perceptron.predict(x)
    print(f"Input: {x}, Prediction: {y_hat}")
```

resulting in:

```
Testing the trained perceptron:
Input: [0, 0], Prediction: 0
Input: [0, 1], Prediction: 1
Input: [1, 0], Prediction: 1
Input: [1, 1], Prediction: 1
```



### With [PyTorch](https://pytorch.org) (and [Autograd](https://pytorch.org/docs/stable/notes/autograd.html))



```python
import torch.nn as nn

class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        # Input layer with 2 inputs, output layer with 1 output
        self.layer         = nn.Linear(2, 1)
        self.step_function = nn.Sigmoid()

    def forward(self, x):
        # Forward pass through the perceptron
        x = self.layer(x)
        x = self.step_function(x)
        return x
```

note - step function for auto gradient backward propagation requires a
step function with derivatives (thus, heaviside with 0/1 changed to sigmoid with values between 0 and 1)


for training use:

``` python
X = torch.tensor([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]], dtype=torch.float32)

y_or  = torch.tensor([[0],  # 0 OR 0 = 0
                      [1],  # 0 OR 1 = 1
                      [1],  # 1 OR 0 = 1
                      [1]], # 1 OR 1 = 1
                       dtype=torch.float32)

model = Perceptron()

# Set up the loss function (binary cross-entropy) and optimizer (Stochastic Gradient Descent)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.1)  # SGD optimizer

# Train the model
epochs = 10000
for epoch in range(epochs):
    optimizer.zero_grad()  # Zero the gradients
    output = model(X) # Forward pass
    loss = criterion(output, y_or)  # Compute the loss
    loss.backward()    # Backward pass (compute gradients)
    optimizer.step()   # Update the weights

# Test the model (after training)
with torch.no_grad():  # No need to compute gradients during testing
    predictions = model(X)
    print("Predictions after training:")
    for i, pred in enumerate(predictions):
        print(f"Input: {X[i].numpy()} -> Prediction: {pred.item()} (Target: {y[i].item()})")
```


resulting in:

```
Predictions after training:
Input: [0. 0.] -> Prediction: 0.02051098830997944 (Target: 0.0)
Input: [0. 1.] -> Prediction: 0.9918055534362793 (Target: 1.0)
Input: [1. 0.] -> Prediction: 0.9918105602264404 (Target: 1.0)
Input: [1. 1.] -> Prediction: 0.9999985694885254 (Target: 1.0)
```



## Multi-Layer Perceptron (2 Inputs, 2 Hiddens, 1 Output)

Train on XOR (non-linear)

Logical XOR Problem:

| Input 1 | Input 2 | Output (XOR) |
|---------|---------|--------------|
|    0    |    0    |       0      |
|    0    |    1    |       1      |
|    1    |    0    |       1      |
|    1    |    1    |       0      |



Let's reverse the samples - PyTorch, NumPy, and Vanilla

PyTorch

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # Input layer has 2 neurons (for 2 inputs), hidden layer has 2 neurons, output has 1 neuron
        self.input_layer   = nn.Linear(2, 2)  # First layer: 2 inputs -> 2 hidden neurons
        self.hidden_layer  = nn.Linear(2, 1)  # Second layer: 2 hidden neurons -> 1 output
        self.act_fn = nn.Sigmoid()            # Sigmoid activation function

    def forward(self, x):
        x = self.act_fn(self.input_layer(x))
        x = self.act_fn(self.hidden_layer(x))
        return x
```


resulting in:

```
Predictions after training:
Input: [0. 0.] -> Prediction: 0.011068678461015224 (Target: 0.0)
Input: [0. 1.] -> Prediction: 0.9913523197174072 (Target: 1.0)
Input: [1. 0.] -> Prediction: 0.9916595816612244 (Target: 1.0)
Input: [1. 1.] -> Prediction: 0.009626330807805061 (Target: 0.0)
```