"""NumPy building blocks for a single neuron and common activations."""

import numpy as np


# -----------------------------------------------------------------------------
# NumPy reference examples
# Uncomment these while experimenting with individual NumPy operations.
# -----------------------------------------------------------------------------
# X = np.array([[1.0, 2.0], [3.0, 4.0]])  # A 2x2 matrix of floats.
# print(X)  # Print the matrix.
#
# print(X.shape)  # Rows and columns: (2, 2).
# print(X[0, 1])  # Access row 0, column 1: 2.0.
#
# z = np.zeros(3)  # Create a 1D array of three zeros.
# print(z)
#
# z = np.array([0.0, 1.0, 2.0])  # Create a 1D array with three elements.
# print(z)
# print(np.exp(z))  # Calculate e**z for every element in the array.


# -----------------------------------------------------------------------------
# Weighted sum: z = X @ w + b
# -----------------------------------------------------------------------------
def weighted_sum_loop(X, w, b):
    """Compute z = X @ w + b using explicit Python loops."""
    n_rows, n_cols = X.shape
    z = np.zeros(n_rows)

    for i in range(n_rows):
        total = 0.0
        for j in range(n_cols):
            total += X[i, j] * w[j]
        z[i] = total + b

    return z


# Keep the original misspelled name so existing code continues to work.
weigted_sum_loop = weighted_sum_loop


def weighted_sum(X, w, b):
    """Compute z = X @ w + b using NumPy matrix multiplication."""
    return X @ w + b


# -----------------------------------------------------------------------------
# Activation functions: y = f(z)
# -----------------------------------------------------------------------------
def identity(z):
    """Return z unchanged."""
    return z


def relu(z):
    """Return max(0, z) element by element."""
    return np.maximum(0, z)


def sigmoid(z):
    """Return 1 / (1 + e**-z) element by element."""
    return 1 / (1 + np.exp(-z))


def tanh(z):
    """Return tanh(z), calculated from exponentials rather than np.tanh."""
    numerator = np.exp(z) - np.exp(-z)
    denominator = np.exp(z) + np.exp(-z)
    return numerator / denominator


def leaky_relu(z, alpha=0.01):
    """Keep positive values; scale negative values by alpha."""
    return np.where(z > 0, z, alpha * z)


def elu(z, alpha=1.0):
    """Return z when positive; otherwise alpha * (e**z - 1)."""
    return np.where(z > 0, z, alpha * (np.exp(z) - 1))


def softplus(z):
    """Return log(1 + e**z), a smooth ReLU-like activation."""
    return np.log(1 + np.exp(z))


def swish(z):
    """Return z * sigmoid(z)."""
    return z * sigmoid(z)


def softmax(z):
    """Convert class scores to probabilities along the final axis."""
    shifted_z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(shifted_z)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


# -----------------------------------------------------------------------------
# Complete single-neuron forward pass
# -----------------------------------------------------------------------------
def forward(X, w, b, activation):
    """Calculate z = X @ w + b, then return y = activation(z)."""
    z = weighted_sum(X, w, b)
    return activation(z)

# -----------------------------------------------------------------------------
# binary cross-entropy (BCE)
# -----------------------------------------------------------------------------
def binary_cross_entropy(y_true, y_pred):
    """epsilon = 1e-15
    This is a very small positive number:
    1e-15 = 0.000000000000001
    
    We use epsilon to prevent values from becoming exactly 0 or 1,
    because log(0) is undefined.
    
    So instead of:
        0 → use epsilon
        1 → use 1 - epsilon
    
    This keeps logarithm calculations safe."""
    epsilon = 1e-15

    """Force every prediction value to stay between epsilon and 1 - epsilon.
    np.clip()
    np.clip(x, a, b) means:
    
        Keep x between the minimum value (a)
        and the maximum value (b).
    
    Mathematically:
    
                 a,  if x < a
    clip(x,a,b)= x,  if a <= x <= b
                 b,  if x > b
    
    In other words:
    
        If x is too small  -> return a
        If x is safe       -> return x unchanged
        If x is too large  -> return b"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    return -np.mean(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )


def train_neuron(X, y_true, w_initial, b_initial, learning_rate=0.1, epochs=1000):
    """Train one sigmoid neuron and return its learned weights and bias.

    This reusable function lets the debug scenarios train on AND, OR, and XOR
    without editing the training loop or changing the source data by hand.
    """
    w = w_initial.copy()
    b = b_initial
    number_of_examples = X.shape[0]

    for _ in range(epochs):
        y_pred = forward(X, w, b, sigmoid)

        dz = y_pred - y_true
        dw = (X.T @ dz) / number_of_examples
        db = np.mean(dz)

        w -= learning_rate * dw
        b -= learning_rate * db

    return w, b

def evaluate_neuron(X, y_true, w, b):
    """Return probabilities, labels, loss, and accuracy for trained parameters."""
    probabilities = forward(X, w, b, sigmoid)
    labels = (probabilities >= 0.5).astype(float)
    loss = binary_cross_entropy(y_true, probabilities)
    accuracy = np.mean(labels == y_true)

    return probabilities, labels, loss, accuracy

# -----------------------------------------------------------------------------
# Learning checks: run only when this file is executed directly.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Inputs, weights, and bias used in the weighted-sum examples.
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    w = np.array([0.5, -1.0])
    b = 0.1

    """choose a learnable binary target. Start with the AND gate
    
    Input       AND output
    [0, 0]  →      0
    [0, 1]  →      0
    [1, 0]  →      0
    [1, 1]  →      1   """
    y_true = np.array([0.0, 0.0, 0.0, 1.0])

    # Loop-based and vectorized weighted sums must agree.
    z_loop = weighted_sum_loop(X, w, b)
    z_fast = weighted_sum(X, w, b)
    print("loop:", z_loop)
    print("fast:", z_fast)
    assert np.allclose(z_loop, z_fast), "loop and @ versions disagree"

    # A batch of four two-input examples.
    X_gate = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    print("gate z:", weighted_sum(X_gate, w, b))

    # Original activation checks.
    print("identity:", identity(7.0))
    print("relu:", relu(-3.0), relu(2.0))
    print("sigmoid:", sigmoid(0.0), sigmoid(2.0), sigmoid(-2.0))
    print("tanh:", tanh(0.0), tanh(2.0), "vs numpy", np.tanh(2.0))
    print("leaky relu:", leaky_relu(-3.0), leaky_relu(2.0))
    print("elu:", elu(-3.0), elu(2.0))
    print("softplus:", softplus(-3.0), softplus(2.0))
    print("swish:", swish(-3.0), swish(2.0))
    print("softmax:", softmax(np.array([-3.0, 2.0])))

    # Forward-pass examples: one function can use different activations.
    print("sigmoid output:", forward(X_gate, w, b, sigmoid))
    print("relu output:", forward(X_gate, w, b, relu))
    print("tanh output:", forward(X_gate, w, b, tanh))

    # Binary cross-entropy loss example.
    y_pred = forward(X_gate, w, b, sigmoid)
    loss = binary_cross_entropy(y_true, y_pred)
    print("predictions:", y_pred)
    print("loss:", loss)

    # -----------------------------------------------------------------------------
    # Implement Gradient Descent for a single neuron
    # -----------------------------------------------------------------------------
    """Error for each training example.
    Positive: prediction was too high.
    Negative: prediction was too low."""
    dz = y_pred - y_true

    # Number of training rows in X_gate.
    # X_gate has shape (4, 2), so this is 4.
    number_of_examples = X_gate.shape[0]

    # Gradient for each weight.
    # X_gate.T groups each input feature across all examples.
    # Dividing by the number of examples gives the average gradient.
    dw = (X_gate.T @ dz) / number_of_examples

    # Gradient for the bias.
    # The bias affects every example equally, so use the average error.
    db = np.mean(dz)

    # Controls how large each learning update is.
    learning_rate = 1.0

    # Move each weight in the direction that reduces loss.
    w_new = w - learning_rate * dw

    # Move the bias in the direction that reduces loss.
    b_new = b - learning_rate * db

    print("dw:", dw)
    print("db:", db)
    print("updated w:", w_new)
    print("updated b:", b_new)

    # Check the new predictions and loss after one gradient descent step.
    new_y_pred = forward(X_gate, w_new, b_new, sigmoid)
    new_loss = binary_cross_entropy(y_true, new_y_pred)

    # Compare the new predictions and loss to the old ones.
    print("new predictions:", new_y_pred)
    print("new loss:", new_loss)

    # Copy initial parameters so we train these copies.
    w_train = w.copy()
    b_train = b

    # Controls the size of each update.
    learning_rate = 0.1

    # Number of complete passes through the training data.
    epochs = 10_000

    for epoch in range(epochs):

        # Forward pass:
        # z = X @ w + b
        # y_pred = sigmoid(z)
        y_pred = forward(X_gate, w_train, b_train, sigmoid)

        # Measure prediction error using Binary Cross-Entropy.
        loss = binary_cross_entropy(y_true, y_pred)

        # Gradient of loss with respect to z.
        # For sigmoid + BCE: dz = y_pred - y_true
        dz = y_pred - y_true

        # Gradient of the weights.
        # X.T @ dz gives the contribution of each weight.
        dw = (X_gate.T @ dz) / number_of_examples

        # Gradient of the bias = average error.
        db = np.mean(dz)

        # Gradient descent: move parameters opposite to the gradient.
        w_train -= learning_rate * dw
        b_train -= learning_rate * db

        # Show loss every 100 epochs.
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss = {loss:.4f}")

    print("trained weights:", w_train)
    print("trained bias:", b_train)
    print("final predictions:", forward(X_gate, w_train, b_train, sigmoid))

    final_probabilities = forward(X_gate, w_train, b_train, sigmoid)

    # Convert probabilities into class lables
    # >= 0.5 becomes 1, < otherwise becomes 0
    final_predictions = (final_probabilities >= 0.5).astype(int)

    # Fraction of predictions that match the expected AND outputs.
    accuracy = np.mean(final_predictions == y_true)

    print("final probabilities:", final_probabilities)
    print("predicted labels:", final_predictions)
    print("true labels:", y_true)
    print(f"accuracy: {accuracy:.0%}")

    # Evaluate the trained neuron
    final_probabilities = forward(X_gate, w_train, b_train, sigmoid)
    final_labels = (final_probabilities >= 0.5).astype(int)
    accuracy = np.mean(final_labels == y_true)
    final_loss = binary_cross_entropy(y_true, final_probabilities)

    print("final loss:", final_loss)
    print("final probabilities:", final_probabilities)
    print("predicted labels:", final_labels)
    print("true labels:", y_true)
    print(f"accuracy: {accuracy:.0%}")

    # -------------------------------------------------------------------------
    # Automated debug scenarios
    # Each scenario has the same four input rows but different expected labels.
    # This makes the model's capabilities and limitations easy to compare.
    # -------------------------------------------------------------------------
    scenario_accuracies = {}
    scenarios = {
        "AND": np.array([0.0, 0.0, 0.0, 1.0]),  # True only for [1, 1].
        "OR": np.array([0.0, 1.0, 1.0, 1.0]),   # False only for [0, 0].
        "XOR": np.array([0.0, 1.0, 1.0, 0.0]),  # True when exactly one input is 1.
    }

    for name, scenario_y_true in scenarios.items():
        # Start every scenario from the same initial weights and bias so the
        # results reflect the dataset, rather than a different initialization.
        w_trained, b_trained = train_neuron(X_gate, scenario_y_true, w, b)

        # Convert the trained neuron's probabilities to labels and measure how
        # closely they match this scenario's expected output.
        probabilities, labels, loss, accuracy = evaluate_neuron(
            X_gate, scenario_y_true, w_trained, b_trained
        )

        print(f"\n{name}")
        print("probabilities:", probabilities)
        print("labels:", labels)
        print(f"loss: {loss:.4f}")
        print(f"accuracy: {accuracy:.0%}")

        # Save the result so assertions below can verify it automatically.
        scenario_accuracies[name] = accuracy

    # Automated tests for one linear sigmoid neuron:
    # AND and OR are linearly separable, so they should be learned perfectly.
    assert scenario_accuracies["AND"] == 1.0
    assert scenario_accuracies["OR"] == 1.0

    # XOR is not linearly separable, so one neuron cannot achieve 100% accuracy.
    assert scenario_accuracies["XOR"] < 1.0