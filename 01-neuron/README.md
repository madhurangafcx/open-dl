# 01 — Single Neuron

## Goal

Build and train a single binary-classification neuron from first principles
with NumPy, then reproduce the same workflow with PyTorch.

The neuron learns the following forward pass:

```text
z = X @ w + b
y = sigmoid(z)
```

Where:

- `X` is a batch of input examples.
- `w` contains one weight for each input feature.
- `b` is the bias.
- `z` is the weighted sum (also called a logit before sigmoid).
- `y` is the final prediction.

For one example, the same calculation is `z = x · w + b`.

## Project structure

```text
01-neuron/
├── README.md
├── src/
│   ├── neuron.py        # NumPy implementation, training, and automated checks
│   └── neuron_torch.py  # PyTorch implementation
└── steps/
    └── step-01-mathematics.md
```

## Learning workflow

| Step | Status | Evidence |
| --- | --- | --- |
| 1. Mathematics | Complete | `steps/step-01-mathematics.md` |
| 2. NumPy implementation | Complete | Loop and vectorized weighted sums |
| 3. Forward pass | Complete | `forward(X, w, b, activation)` |
| 4. Loss | Complete | Binary cross-entropy (BCE) |
| 5. Derive gradients | Complete | `dz`, `dw`, and `db` |
| 6. Backpropagation | Complete | Gradient-descent parameter updates |
| 7. Train model | Complete | Sigmoid neuron trained on AND |
| 8. Debug and analyze | Complete | Automated AND, OR, and XOR scenarios |
| 9. PyTorch implementation | Complete | `nn.Linear`, autograd, and SGD |
| 10. Compare results | Complete | NumPy and PyTorch both learn AND |

## NumPy implementation

`src/neuron.py` implements the following building blocks.

### Weighted sum

Two equivalent implementations verify the underlying matrix operation:

```python
weighted_sum_loop(X, w, b)  # Explicit Python loops
weighted_sum(X, w, b)       # Vectorized NumPy: X @ w + b
```

The script asserts that both approaches produce the same values.

### Activations

The project includes the following activations:

- identity
- ReLU
- sigmoid
- tanh
- leaky ReLU
- ELU
- softplus
- swish
- softmax

The binary classifier uses sigmoid because it maps logits to values between
`0` and `1`.

### Loss and gradients

For binary labels, the model uses binary cross-entropy:

```text
BCE = -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
```

With a sigmoid output and BCE loss, the gradients are:

```text
dz = y_pred - y_true
dw = (X.T @ dz) / number_of_examples
db = mean(dz)
```

Parameters are updated with gradient descent:

```text
w = w - learning_rate * dw
b = b - learning_rate * db
```

## Debug scenarios and results

The NumPy script automatically trains and evaluates the same neuron on three
logic-gate datasets:

| Scenario | Targets | Expected result for one neuron |
| --- | --- | --- |
| AND | `[0, 0, 0, 1]` | Learns perfectly |
| OR | `[0, 1, 1, 1]` | Learns perfectly |
| XOR | `[0, 1, 1, 0]` | Cannot learn perfectly |

The automated checks assert that AND and OR reach `100%` accuracy, while XOR
remains below `100%` accuracy. XOR fails because one neuron creates only one
linear decision boundary; XOR is not linearly separable.

For the configured AND training run, the NumPy model reaches `100%` accuracy
and drives the loss down to approximately `0.0175` after 10,000 epochs.

## PyTorch implementation

`src/neuron_torch.py` recreates the AND classifier using PyTorch:

```python
model = nn.Linear(in_features=2, out_features=1)
loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```

The PyTorch training loop performs the same work as the manual NumPy loop:

| NumPy | PyTorch |
| --- | --- |
| `X @ w + b` | `model(X)` |
| `sigmoid(z)` | `torch.sigmoid(logits)` |
| `binary_cross_entropy(...)` | `BCEWithLogitsLoss()` |
| Manual `dz`, `dw`, `db` | `loss.backward()` |
| Manual parameter updates | `optimizer.step()` |

`BCEWithLogitsLoss` receives raw logits and applies sigmoid internally in a
numerically stable way. Sigmoid is applied separately only when displaying
final probabilities.

The PyTorch model also reaches `100%` accuracy on the AND dataset.

## Run the project

From the `01-neuron` directory:

```bash
/opt/anaconda3/bin/python src/neuron.py
python src/neuron_torch.py
```

The second command requires PyTorch in the active Python environment:

```bash
python -m pip install torch torchvision
```

## Key takeaways

- A neuron first computes a weighted sum, then applies an activation.
- Sigmoid plus BCE is suitable for binary classification.
- Backpropagation calculates how each parameter contributed to loss.
- Gradient descent uses those gradients to improve weights and bias.
- AND and OR are linearly separable; XOR requires a network with a hidden
  layer.
- PyTorch automates gradient calculation and parameter updates, while the
  underlying mathematics remains the same as the NumPy implementation.

## Next step

Build a two-layer neural network that learns XOR:

```text
2 inputs → hidden layer → sigmoid output
```

