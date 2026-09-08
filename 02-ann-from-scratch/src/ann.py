"""Two-layer Artificial Neural Network (ANN) built from scratch with NumPy.

Architecture:
  Input (2 features) -> Dense Layer 1 -> ReLU -> Dense Layer 2 -> Softmax -> Output (2 classes)

Strictly mirrors each mathematical section in steps/step-01-mathematics.md:
  Q1-Q4: Network architecture and parameter shapes
  Q5:    Forward pass (Z1 -> A1 -> Z2 -> P)
  Q6:    Categorical cross-entropy loss (L)
  Q7:    Backpropagation (dZ2 -> dW2, db2 -> dA1 -> dZ1 -> dW1, db1)
  Q8:    Gradient descent parameter updates
  Q9:    He / Xavier parameter initialization
  Q10:   Prediction, evaluation, and training
"""

import numpy as np


# =============================================================================
# Q4 & Q5 — Forward pass building blocks
# =============================================================================
def dense_forward(X, W, b):
    """Dense layer forward pass: Z = X @ W + b.

    X: (m, n_in)          m examples, one per row
    W: (n_in, n_out)      weights connecting inputs to outputs
    b: (1, n_out)         one bias per neuron, broadcasted over m rows
    returns (m, n_out)    pre-activations (logits)
    """
    return X @ W + b


def dense_forward_batch_loop(X, W, b):
    """Identical result to dense_forward, computed with explicit loops (oracle proof).

    Proves that matrix multiplication (@) is nothing more than three nested loops:
      i over examples, j over neurons, k over input features.
    """
    m, n_in = X.shape
    n_out = W.shape[1]
    assert n_in == W.shape[0], f"X features ({n_in}) != W rows ({W.shape[0]})"

    Z = np.zeros((m, n_out))
    for i in range(m):
        for j in range(n_out):
            total = 0.0
            for k in range(n_in):
                total += X[i, k] * W[k, j]
            Z[i, j] = total + b[0, j]
    return Z


def relu(Z):
    """ReLU activation: A = max(0, Z) element by element."""
    return np.maximum(0.0, Z)


def relu_derivative(Z):
    """Derivative of ReLU: 1 if Z > 0, else 0."""
    return (Z > 0.0).astype(float)


def softmax(Z):
    """Numerically stable softmax along the last axis.

    Subtracting the row maximum prevents overflow without altering probabilities:
      P_i = exp(Z_i - max(Z)) / sum(exp(Z_j - max(Z)))

    Works seamlessly on single vectors (n_classes,) or batches (m, n_classes).
    """
    shifted = Z - np.max(Z, axis=-1, keepdims=True)
    exp_Z = np.exp(shifted)
    return exp_Z / np.sum(exp_Z, axis=-1, keepdims=True)


# =============================================================================
# Q6 — Softmax categorical cross-entropy loss
# =============================================================================
def cross_entropy(Y, P):
    """Categorical cross-entropy loss: L = -(1/m) * sum(sum(Y * log(P))).

    Y: (m, n_classes) one-hot target matrix
    P: (m, n_classes) softmax predicted probabilities
    returns scalar mean loss across the batch
    """
    eps = 1e-15  # Prevent log(0) = -inf and 0 * -inf = NaN
    P_clipped = np.clip(P, eps, 1.0 - eps)

    if Y.ndim == 1:
        return -np.sum(Y * np.log(P_clipped))

    losses = -np.sum(Y * np.log(P_clipped), axis=1)
    return np.mean(losses)


# =============================================================================
# Q9 — Parameter initialization (Symmetry breaking)
# =============================================================================
def init_parameters(n_in=2, n_hidden=4, n_out=2, seed=None):
    """Initialize network weights and biases.

    - W1: He (Kaiming) initialization for ReLU -> N(0, 2 / n_in)
    - b1: Zeros (1, n_hidden)
    - W2: Xavier (Glorot) initialization for Softmax -> N(0, 2 / (n_hidden + n_out))
    - b2: Zeros (1, n_out)
    """
    if seed is not None:
        np.random.seed(seed)

    W1 = np.random.randn(n_in, n_hidden) * np.sqrt(2.0 / n_in)
    b1 = np.zeros((1, n_hidden))

    W2 = np.random.randn(n_hidden, n_out) * np.sqrt(2.0 / (n_hidden + n_out))
    b2 = np.zeros((1, n_out))

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


# =============================================================================
# Complete Model Functions: Forward, Backward, Update, Train, Predict
# =============================================================================
def forward(X, params):
    """Complete forward pass through the two-layer network.

    X -> Z1 -> A1 -> Z2 -> P
    Returns:
      P:     Predicted softmax probabilities (m, n_out)
      cache: Intermediate activations needed for backpropagation
    """
    # Ensure X is 2D batch format
    if X.ndim == 1:
        X = X.reshape(1, -1)

    Z1 = dense_forward(X, params["W1"], params["b1"])
    A1 = relu(Z1)
    Z2 = dense_forward(A1, params["W2"], params["b2"])
    P = softmax(Z2)

    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "P": P}
    return P, cache


def backward(Y, cache, params):
    """Complete backward pass (Backpropagation) derived in Q7.

    Equations:
      dZ2 = (P - Y) / m
      dW2 = A1.T @ dZ2
      db2 = sum(dZ2, axis=0, keepdims=True)
      dA1 = dZ2 @ W2.T
      dZ1 = dA1 * (Z1 > 0)
      dW1 = X.T @ dZ1
      db1 = sum(dZ1, axis=0, keepdims=True)
    """
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)

    m = Y.shape[0]
    X = cache["X"]
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    P = cache["P"]
    W2 = params["W2"]

    # 1. Output-layer gradients
    dZ2 = (P - Y) / m
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    # 2. Propagate error to hidden layer
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0.0)  # ReLU derivative

    # 3. Hidden-layer parameter gradients
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return {
        "dZ2": dZ2,
        "dW2": dW2,
        "db2": db2,
        "dA1": dA1,
        "dZ1": dZ1,
        "dW1": dW1,
        "db1": db1,
    }


def update_parameters(params, grads, learning_rate):
    """Gradient descent parameter update: theta = theta - lr * dtheta (Q8)."""
    params["W1"] -= learning_rate * grads["dW1"]
    params["b1"] -= learning_rate * grads["db1"]
    params["W2"] -= learning_rate * grads["dW2"]
    params["b2"] -= learning_rate * grads["db2"]
    return params


def predict(X, params):
    """Predict class labels: y_hat = argmax(P, axis=1) (Q10)."""
    P, _ = forward(X, params)
    return np.argmax(P, axis=1)


def evaluate(X, Y, params):
    """Evaluate network probabilities, predictions, loss, and accuracy (Q10)."""
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)

    P, _ = forward(X, params)
    predictions = np.argmax(P, axis=1)
    true_labels = np.argmax(Y, axis=1)

    loss = cross_entropy(Y, P)
    accuracy = np.mean(predictions == true_labels)

    return P, predictions, loss, accuracy


def train(X, Y, params, learning_rate=1.0, epochs=2000, verbose=False):
    """Train the two-layer ANN on batch data via full-batch gradient descent.

    Returns:
      params:       Trained parameters
      loss_history: Recorded loss per epoch
    """
    loss_history = []

    for epoch in range(epochs):
        # 1. Forward pass
        P, cache = forward(X, params)

        # 2. Compute loss
        loss = cross_entropy(Y, P)
        loss_history.append(loss)

        # 3. Backpropagation
        grads = backward(Y, cache, params)

        # 4. Gradient descent update
        params = update_parameters(params, grads, learning_rate)

        if verbose and (epoch % 200 == 0 or epoch == epochs - 1):
            preds = np.argmax(P, axis=1)
            acc = np.mean(preds == np.argmax(Y, axis=1))
            print(f"Epoch {epoch:4d} | Loss: {loss:.6f} | Accuracy: {acc * 100:.1f}%")

    return params, loss_history


# =============================================================================
# Verification suite matching steps/step-01-mathematics.md exactly
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("STEP 1: VERIFYING MATHEMATICAL EQUATIONS FROM step-01-mathematics.md")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Q4 — Illustrative parameters from the mathematics document
    # -------------------------------------------------------------------------
    W1 = np.array([
        [ 0.5, -0.5,  1.0, -1.0],
        [ 1.0,  0.5, -1.0, -0.5],
    ])  # shape (2, 4)
    b1 = np.array([[0.1, 0.1, 0.1, 0.1]])  # shape (1, 4)

    W2 = np.array([
        [-0.5,  0.5],
        [-1.0,  1.0],
        [ 0.5, -0.5],
        [-0.5,  0.5],
    ])  # shape (4, 2)
    b2 = np.array([[0.1, -0.1]])  # shape (1, 2)

    illustrative_params = {"W1": W1.copy(), "b1": b1.copy(), "W2": W2.copy(), "b2": b2.copy()}

    # -------------------------------------------------------------------------
    # Q5 — Single-example forward pass check (x = [0, 1] -> class 1)
    # -------------------------------------------------------------------------
    x_single = np.array([0.0, 1.0])
    p_single, cache_single = forward(x_single, illustrative_params)

    print("\n--- Q5: Single Example Check [0, 1] ---")
    print("Z1:", cache_single["Z1"][0])
    print("A1:", cache_single["A1"][0])
    print("Z2:", cache_single["Z2"][0])
    print("P: ", p_single[0])

    assert np.allclose(cache_single["Z1"][0], [1.1, 0.6, -0.9, -0.4])
    assert np.allclose(cache_single["A1"][0], [1.1, 0.6, 0.0, 0.0])
    assert np.allclose(cache_single["Z2"][0], [-1.05, 1.05])
    assert np.allclose(p_single[0], [0.10909682, 0.89090318])
    print("✓ Single-example calculations match step-01-mathematics.md exactly.")

    # -------------------------------------------------------------------------
    # Q5 — Full batch forward pass check (All 4 XOR examples)
    # -------------------------------------------------------------------------
    X = np.array([
        [0.0, 0.0],  # example 0 -> class 0
        [0.0, 1.0],  # example 1 -> class 1
        [1.0, 0.0],  # example 2 -> class 1
        [1.0, 1.0],  # example 3 -> class 0
    ])

    Y = np.array([
        [1.0, 0.0],  # example 0
        [0.0, 1.0],  # example 1
        [0.0, 1.0],  # example 2
        [1.0, 0.0],  # example 3
    ])

    # Prove matrix multiplication equals triple loop
    Z1_loop = dense_forward_batch_loop(X, illustrative_params["W1"], illustrative_params["b1"])
    P_batch, cache_batch = forward(X, illustrative_params)

    print("\n--- Q5: Full Batch Forward Pass Check ---")
    print("Z1:\n", cache_batch["Z1"])
    print("A1:\n", cache_batch["A1"])
    print("Z2:\n", cache_batch["Z2"])
    print("P:\n", P_batch)

    assert np.allclose(Z1_loop, cache_batch["Z1"])
    assert np.allclose(cache_batch["Z1"], [
        [0.1, 0.1, 0.1, 0.1],
        [1.1, 0.6, -0.9, -0.4],
        [0.6, -0.4, 1.1, -0.9],
        [1.6, 0.1, 0.1, -1.4],
    ])
    assert np.allclose(cache_batch["A1"], [
        [0.1, 0.1, 0.1, 0.1],
        [1.1, 0.6, 0.0, 0.0],
        [0.6, 0.0, 1.1, 0.0],
        [1.6, 0.1, 0.1, 0.0],
    ])
    assert np.allclose(cache_batch["Z2"], [
        [-0.05,  0.05],
        [-1.05,  1.05],
        [ 0.35, -0.35],
        [-0.75,  0.75],
    ])
    assert np.allclose(P_batch, [
        [0.47502081, 0.52497919],
        [0.10909682, 0.89090318],
        [0.66818777, 0.33181223],
        [0.18242552, 0.81757448],
    ])
    assert np.allclose(np.sum(P_batch, axis=1), np.ones(4))
    print("✓ Full batch forward pass matches step-01-mathematics.md exactly.")

    # -------------------------------------------------------------------------
    # Q6 — Categorical cross-entropy loss check
    # -------------------------------------------------------------------------
    loss = cross_entropy(Y, P_batch)
    print("\n--- Q6: Cross-Entropy Loss Check ---")
    print(f"Batch Loss: {loss:.8f}")
    assert np.isclose(loss, 0.91612888, atol=1e-6)
    print("✓ Loss calculation matches step-01-mathematics.md exactly.")

    # -------------------------------------------------------------------------
    # Q7 — Backpropagation gradients check
    # -------------------------------------------------------------------------
    grads = backward(Y, cache_batch, illustrative_params)

    print("\n--- Q7: Backpropagation Gradients Check ---")
    print("dZ2:\n", grads["dZ2"])
    print("dW2:\n", grads["dW2"])
    print("db2:\n", grads["db2"])
    print("dA1:\n", grads["dA1"])
    print("dZ1:\n", grads["dZ1"])
    print("dW1:\n", grads["dW1"])
    print("db1:\n", grads["db1"])

    assert np.allclose(grads["dZ2"], [
        [-0.13124480,  0.13124480],
        [ 0.02727421, -0.02727421],
        [ 0.16704694, -0.16704694],
        [-0.20439362,  0.20439362],
    ], atol=1e-5)
    assert np.allclose(grads["dW2"], [
        [-0.20992448,  0.20992448],
        [-0.01719932,  0.01719932],
        [ 0.15018780, -0.15018780],
        [-0.01312448,  0.01312448],
    ], atol=1e-5)
    assert np.allclose(grads["db2"], [[-0.14131727, 0.14131727]], atol=1e-5)
    assert np.allclose(grads["dA1"], [
        [ 0.13124480,  0.26248959, -0.13124480,  0.13124480],
        [-0.02727421, -0.05454841,  0.02727421, -0.02727421],
        [-0.16704694, -0.33409389,  0.16704694, -0.16704694],
        [ 0.20439362,  0.40878724, -0.20439362,  0.20439362],
    ], atol=1e-5)
    assert np.allclose(grads["dZ1"], [
        [ 0.13124480,  0.26248959, -0.13124480,  0.13124480],
        [-0.02727421, -0.05454841,  0.00000000,  0.00000000],
        [-0.16704694,  0.00000000,  0.16704694,  0.00000000],
        [ 0.20439362,  0.40878724, -0.20439362,  0.00000000],
    ], atol=1e-5)
    assert np.allclose(grads["dW1"], [
        [0.03734668, 0.40878724, -0.03734668, 0.0],
        [0.17711941, 0.35423883, -0.20439362, 0.0],
    ], atol=1e-5)
    assert np.allclose(grads["db1"], [[0.14131727, 0.61672842, -0.16859147, 0.13124480]], atol=1e-5)
    print("✓ Backpropagation gradients match step-01-mathematics.md exactly.")

    # -------------------------------------------------------------------------
    # Q8, Q9, Q10 — Full training run on XOR with random He initialization
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: TRAINING THE ANN ON XOR FROM FIRST PRINCIPLES")
    print("=" * 70)

    trainable_params = init_parameters(n_in=2, n_hidden=4, n_out=2, seed=42)
    initial_loss = cross_entropy(Y, forward(X, trainable_params)[0])
    print(f"Initial loss before training: {initial_loss:.6f}")

    trained_params, loss_history = train(
        X, Y, trainable_params, learning_rate=1.0, epochs=2000, verbose=True
    )

    final_P, final_preds, final_loss, final_acc = evaluate(X, Y, trained_params)

    print("\n--- XOR Truth Table Results ---")
    for i in range(4):
        x_in = X[i]
        expected_class = np.argmax(Y[i])
        pred_class = final_preds[i]
        prob = final_P[i, pred_class]
        print(f"Input: {x_in} | Expected: {expected_class} | Predicted: {pred_class} (p={prob:.4f})")

    print(f"\nFinal Loss:     {final_loss:.6f}")
    print(f"Final Accuracy: {final_acc * 100:.1f}%")

    assert final_acc == 1.0, "ANN failed to solve XOR!"
    print("\n✓ SUCCESS: Two-layer ANN successfully learned the XOR non-linear function!")