import torch
import torch.nn as nn
import torch.optim as optim



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

# Define the dataset for Logical OR
# X contains inputs, y contains the target output
X = torch.tensor([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])  # Input data

y = torch.tensor([[0.0],  # 0 OR 0 = 0
                  [1.0],  # 0 OR 1 = 1
                  [1.0],  # 1 OR 0 = 1
                  [1.0]])  # 1 OR 1 = 1

# Initialize the model
model = Perceptron()

# Set up the loss function (binary cross-entropy) and optimizer (Stochastic Gradient Descent)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.1)  # SGD optimizer

# Train the model
epochs = 10000
for epoch in range(epochs):
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    output = model(X)

    # Compute the loss
    loss = criterion(output, y)

    # Backward pass (compute gradients)
    loss.backward()

    # Update the weights
    optimizer.step()

    # Print the loss every epoch for monitoring
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Test the model (after training)
with torch.no_grad():  # No need to compute gradients during testing
    predictions = model(X)
    ## predictions = predictions.round()  # Round the output to 0 or 1
    print("\nPredictions after training:")
    for i, pred in enumerate(predictions):
        print(f"Input: {X[i].numpy()} -> Prediction: {pred.item()} (Target: {y[i].item()})")
