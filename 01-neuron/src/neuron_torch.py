# -----------------------------------------------------------------------------
# Input X
#    ↓
# Linear neuron: X @ w + b
#    ↓
# Logits
#    ↓
# BCEWithLogitsLoss
#    ↓
# loss.backward()
#    ↓
# Calculate gradients
#    ↓
# optimizer.step()
#    ↓
# Update w and b
#    ↓
# Repeat
# -----------------------------------------------------------------------------
# nn.Linear()          → creates the neuron
# BCEWithLogitsLoss()  → calculates the loss
# zero_grad()          → clears old gradients
# backward()           → calculates new gradients
# step()               → updates weights/bias
# sigmoid()            → converts logits to probabilities
# -----------------------------------------------------------------------------
# |      NumPy code          |       PyTorch         |
# | ------------------------ | --------------------- |
# | `X @ w + b`              | `model(X)`            |
# | `sigmoid(z)`             | `torch.sigmoid()`     |
# | `binary_cross_entropy()` | `BCEWithLogitsLoss()` |
# | `dz`, `dw`, `db`         | `loss.backward()`     |
# | `w -= lr * dw`           | `optimizer.step()`    |
# | `b -= lr * db`           | `optimizer.step()`    |


# torch gives you tensors and mathematical operations.
import torch

# torch.nn gives you neural-network building blocks.
import torch.nn as nn

# Make the initial random weights repeatble for reproducibility
# PyTorch normally initializes the weights randomly.
# For example, it might randomly create:
# w=[0.34,−0.82]
torch.manual_seed(42)

# Same AND-gate dataset used by the NumPy implmentation
# Shape:
#     (4,2)
# Meaning:
#     4 examples
#     2 features per example
X_gate = torch.tensor(
    [
        [0, 0], 
        [0, 1], 
        [1, 0], 
        [1, 1]
    ], dtype=torch.float32
)

# Shape:
#     (4,2)
# The correspondence is:
# [0, 0] → 0
# [0, 1] → 0
# [1, 0] → 0
# [1, 1] → 1
y_true = torch.tensor(
    [
        [0], 
        [0], 
        [0], 
        [1]
    ], dtype=torch.float32
)

# One neuron: 2 inputs --> 1 output, including a bias term
# Create one neuron with 2 inputs and 1 output.
# Internally: z = X @ w + b
model = nn.Linear(in_features=2, out_features=1)

# BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class. 
# This version is more numerically stable than using a plain Sigmoid followed by 
# a BCELoss as, by combining the operations into one layer, we take advantage of 
# the log-sum-exp trick for numerical stability.
loss_function = nn.BCEWithLogitsLoss()

# Pytorch will update the model's weights and bias for us
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epochs = 1000

for epoch in range(epochs):
    # Forward pass: produces raw scores called logits
    # logits = X @ w + b
    logits = model(X_gate)

    # Calculate how wrong the predictions are.
    loss = loss_function(logits, y_true)

    # Remove gradients from the previous iteration.
    optimizer.zero_grad()

    # Backpropagation: calculate gradients of the loss.
    loss.backward()

    # Update weights and bias using the calculated gradients.
    optimizer.step()

    # Print the loss every 100 epochs.
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")

    # Convert final logits to probabilities and then binary labels

# Disable gradient calculation because training is finished.
with torch.no_grad():

    # Convert logits into probabilities using sigmoid.
    probabilities = torch.sigmoid(model(X_gate))

    # Convert probabilities into binary labels using 0.5 as the threshold.
    predictions = (probabilities > 0.5).float()

    # Calculate the percentage of predictions that are correct.
    accuracy = (predictions == y_true).float().mean()

print("weights:", model.weight)
print("bias:", model.bias)
print("probabilities:", probabilities.squeeze())
print("labels:", predictions.squeeze())
print("true labels:", y_true.squeeze())
print(f"accuracy: {accuracy.item():.0%}")
