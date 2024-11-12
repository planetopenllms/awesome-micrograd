import numpy as np


def heaviside(x):
    return 1 if x > 0 else 0


class Perceptron:
    def __init__(self):
        self.w = np.zeros(2)
        self.b = 0
        self.step_function = heaviside

    def predict(self, x):
        total = np.dot(x, self.w) + self.b
        return self.step_function(total)

    def train(self, X, Y, epochs=10, learning_rate=0.1):
        X = np.array( X )
        Y = np.array( Y )
        for epoch in range(epochs):
            for x, y in zip(X, Y):
                y_hat = self.predict(x)
                error = y - y_hat
                # Update weights and bias based on the error
                self.w += learning_rate * error * x
                self.b += learning_rate * error
            # Optional: print the error during training
            print(f'Epoch {epoch+1}/{epochs} - Weights: {self.w}, Bias: {self.b}')

# Define the training data for logical OR
X_train = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

Y_or = [0, 1, 1, 1]   # OR
# Y_xor = [0, 0, 0, 1]   # XOR

# Initialize the perceptron
perceptron = Perceptron()

# Train the perceptron on the logical OR data
perceptron.train(X_train, Y_or)

# Test the trained perceptron
print("\nTesting the trained perceptron:")
for x in X_train:
    y_hat = perceptron.predict(x)
    print(f"Input: {x}, Prediction: {y_hat}")
