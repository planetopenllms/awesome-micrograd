import torch
import torch.nn as nn
import torch.optim as optim


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

# 2. Prepare the XOR dataset (4 examples, each with 2 inputs)
X = torch.tensor([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])
y = torch.tensor([[0.0],
                  [1.0],
                  [1.0],
                  [0.0]])

# 3. Initialize the model, loss function and optimizer
model = MLP()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. Train the model
num_epochs = 20000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)

    # Calculate loss
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()  # Zero the gradients
    loss.backward()  # Backpropagate the loss
    optimizer.step()  # Update the weights

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. Test the model
with torch.no_grad():
    predictions = model(X)
    # predicted = (predicted > 0.5).float()  # Convert output probabilities to 0 or 1
    print("\nPredictions after training:")
    for i, pred in enumerate(predictions):
        print(f"Input: {X[i].numpy()} -> Prediction: {pred.item()} (Target: {y[i].item()})")
