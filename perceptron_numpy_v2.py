import numpy as np

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)



# Define the Multi-Layer Perceptron class
class MLP:
    def __init__(self):
        # Initialize the weights for the input layer and hidden layer
        self.weights_input_hidden = np.random.randn(2, 2)  # weights between input and hidden layer
        self.weights_hidden_output = np.random.randn(2, 1)  # weights between hidden and output layer
        # Initialize biases for hidden and output layers
        self.bias_hidden = np.random.randn(1, 2)
        self.bias_output = np.random.randn(1, 1)

    def forward(self, X):
        # Input to hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)  # Apply sigmoid activation function

        # Hidden to output layer
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_input)  # Apply sigmoid activation function

        return self.output

    def backward(self, X, y):
        # Calculate output layer error
        output_error = y - self.output
        output_delta = output_error * sigmoid_derivative(self.output)

        # Calculate hidden layer error
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate

        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs=10000, learning_rate=0.1):
        self.learning_rate = learning_rate
        for epoch in range(epochs):
            self.forward(X)  # Forward pass
            self.backward(X, y)  # Backward pass (update weights and biases)

            # Print error every 1000 epochs
            if epoch % 1000 == 0:
                error = np.mean(np.abs(y - self.output))
                print(f"Epoch {epoch}/{epochs} - Error: {error}")

    def predict(self, X):
        return self.forward(X)

# Define the training data for XOR
X = np.array([
    [0, 0],  # Input: [0, 0]
    [0, 1],  # Input: [0, 1]
    [1, 0],  # Input: [1, 0]
    [1, 1]   # Input: [1, 1]
])

y = np.array([
    [0],  # Output: XOR([0, 0]) = 0
    [1],  # Output: XOR([0, 1]) = 1
    [1],  # Output: XOR([1, 0]) = 1
    [0]   # Output: XOR([1, 1]) = 0
])

# Initialize and train the MLP
mlp = MLP()
mlp.train(X, y, epochs=20000, learning_rate=0.1)

# Test the trained MLP
print("\nTesting the trained MLP:")
for x in X:
    prediction = mlp.predict(np.array([x]))  # Predict for each input
    print(f"Input: {x}, Prediction: {prediction[0][0]}")
