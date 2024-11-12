# Awesome ?erceptrons

An awesome collection of perceptrons - single-layer (OR/AND) and multi-layer (XOR)



## Single-Layer Perceptron (2 Inputs, 1 Output)


```python
def step_function(x):
    return 1 if x > 0 else 0
```

note - for step function the error is 1 / -1 or 0 for bingo! no error.




Vanilla

```python
class Perceptron:
    # initialize w(eights) and b(ias)
    def __init__(self):
        self.w = [0,0]
        self.b =  0

    def predict(self, x):
        total = self.w[0] * x[0] +\
                self.w[1] * x[1] + self.b
        return step_function(total)

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


With NumPy


```python
import numpy as np

class Perceptron:
    def __init__(self):
        self.w = np.zeros(2)
        self.b = 0

    def predict(self, x):
        total = np.dot(x, self.w) + self.b
        return step_function(total)

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

Y_train_or   = [0, 1, 1, 1]  # Output labels for OR operation
# Y_and  = [0, 0, 0, 1]  # Output labels for AND operation
# Y_xor  = [0, 1, 1, 0]  # Output labels for XOR operation

perceptron = Perceptron()
perceptron.train(X_train, Y_train_or)

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


