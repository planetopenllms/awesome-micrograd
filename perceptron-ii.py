# Initialize weights and bias
#   with random numbers -1 to 1  e.g. 2*random.random()-1
#   or try/use random.random() for 0 to 1

import random


# Step activation function: Outputs 1 if x > 0 else 0
def heaviside(x):
    return 1 if x > 0 else 0

class Perceptron:
    def __init__(self):
        self.w = [2*random.random()-1,
                  2*random.random()-1]
        self.b =  2*random.random()-1
        self.step_function = heaviside

    def predict(self, x):
        total = self.w[0] * x[0] +\
                self.w[1] * x[1] + self.b
        return self.step_function(total)

    # Train the perceptron using the perceptron learning rule
    def train(self, X, Y, epochs=100, learning_rate=0.1):
        for epoch in range(epochs):
            for x, y in zip(X, Y):
                y_hat = self.predict(x)
                # error is 0 or 1 or -1
                error = y - y_hat
                if error != 0:
                    print( f'  error on {x}: {error} - expected {y}; got {y_hat}' )
                # update weights and bias based on the error
                self.w[0] += learning_rate * error * x[0]
                self.w[1] += learning_rate * error * x[1]
                self.b    += learning_rate * error
            # print the error after every epoch
            print(f'Epoch {epoch + 1}/{epochs} - Weights: {self.w}, Bias: {self.b}')


# Define the training data for logical OR
X_train = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

Y_or  = [0, 1, 1, 1]  # Output labels for OR operation
Y_and = [0, 0, 0, 1]  # Output labels for AND operation
Y_xor = [0, 1, 1, 0]  # Output labels for XOR operation

# Initialize the perceptron
perceptron = Perceptron()  # Two inputs for logical OR

# Train the perceptron on the logical OR data
perceptron.train( X_train, Y_or )

# Test the trained perceptron
print("\nTesting the trained perceptron:")
for x in X_train:
    y_hat = perceptron.predict(x)
    print(f"Input: {x}, Prediction: {y_hat}")
