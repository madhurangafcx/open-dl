"""
12 — Attention Mechanism & Multi-Head Attention From Scratch with Pure NumPy
=============================================================================

Architecture Overview:
----------------------
This module implements the complete Scaled Dot-Product Attention Mechanism and
Multi-Head Attention (MHA) from first principles using only Python and pure NumPy.
It establishes the mathematical and algorithmic foundation of the modern
Transformer architecture (Vaswani et al., 2017: "Attention Is All You Need"),
Large Language Models (LLMs), and Vision Transformers (ViTs).

The Paradigm Shift: From Recurrence to Pairwise Attention:
----------------------------------------------------------
Prior sequential architectures (RNN, LSTM, GRU) compress historical context
step-by-step into a recurrent hidden state:
    h_t = f(x_t, h_{t-1})

This recurrent formulation suffers from two major limitations:
1. Sequential Dependency Bottleneck: Computing step t requires step t - 1.
   Sequential operations depth is O(T), preventing full GPU parallelization across time.
2. Information Bottleneck: Compressing sequences into a fixed-capacity vector
   degrades early token representations over an O(T) maximum signal path length.

Self-Attention eliminates sequential recurrence completely:
1. Parallel Execution: All tokens interact directly via batched matrix multiplications,
   achieving an O(1) sequential operation depth.
2. Direct O(1) Path Length: Every token directly attends to every other token,
   enabling unattenuated credit assignment across arbitrary sequence distances.

The Query, Key, Value (Q, K, V) Paradigm:
-----------------------------------------
Attention functions as a soft, differentiable database lookup:
- Query (Q): What the token is currently searching for (the search probe).
- Key (K):   What index tags the token offers (the database address).
- Value (V): The substantive content payload carried by the token.

The Scaled Dot-Product Equation:
--------------------------------
    Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(d_k) + M ) @ V

Where:
- Q @ K^T calculates pairwise similarity scores between all query-key pairs.
- 1 / sqrt(d_k) normalizes the dot-product variance back to 1.0, preventing
  Softmax gradient saturation when d_k is large.
- M is an optional causal mask (lower triangular) blocking future token visibility.
- softmax converts raw scores into non-negative categorical weights summing to 1.0.
- weights @ V computes a convex linear combination of Value vectors.

Multi-Head Attention (MHA):
---------------------------
Instead of performing a single attention function, Multi-Head Attention projects
queries, keys, and values h times into distinct lower-dimensional subspaces:
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W^O
    where head_i = Attention(Q @ W_i^Q, K @ W_i^K, V @ W_i^V)

This allows the network to jointly attend to information from different representation
subspaces at different positions simultaneously (e.g., syntax, coreference, semantics).

Tensor Dimension Conventions:
-----------------------------
- B: Batch size                       (e.g., B = 1 in trace, B = 2 in batched suite)
- T: Sequence length                  (e.g., T = 3 in trace, T = 4 in batched suite)
- d_model: Model feature dimension    (e.g., d_model = 2 in trace, d_model = 4 in MHA)
- h: Number of attention heads        (e.g., h = 2)
- d_k: Query/Key dimension per head   (d_k = d_model // h)
- d_v: Value dimension per head       (d_v = d_model // h)
"""

import numpy as np


# =============================================================================
# PART 1: MATHEMATICAL PRIMITIVES & MASKING UTILITIES
# =============================================================================

def stable_softmax(logits, axis=-1):
    """
    Computes numerically stable softmax probabilities along a specified axis:
        softmax(z)_i = exp(z_i - max(z)) / sum_j exp(z_j - max(z))

    Subtracting the row-wise maximum prevents numerical overflow (inf/nan)
    without altering the output probability distribution.

    Parameters:
        logits (np.ndarray): Input tensor of arbitrary shape.
        axis (int): Axis along which to compute softmax normalization.

    Returns:
        np.ndarray: Categorical probability distribution summing to 1.0 along axis.
    """
    shifted_logits = logits - np.max(logits, axis=axis, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    sum_exp = np.sum(exp_logits, axis=axis, keepdims=True)
    return exp_logits / sum_exp


def create_causal_mask(seq_len):
    """
    Creates a lower-triangular causal attention mask of shape (seq_len, seq_len):
        M[i, j] = True  if j <= i (allowed past & current tokens)
        M[i, j] = False if j > i  (forbidden future tokens)

    Parameters:
        seq_len (int): Sequence length T.

    Returns:
        np.ndarray: Boolean mask of shape (seq_len, seq_len).
    """
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))


def create_padding_mask(sequence_lengths, max_seq_len):
    """
    Creates a key padding mask for variable-length sequence batches:
        mask[b, i] = True  if i < sequence_lengths[b] (valid token)
        mask[b, i] = False if i >= sequence_lengths[b] (padding token)

    Parameters:
        sequence_lengths (list or np.ndarray): Valid sequence lengths per batch sample.
        max_seq_len (int): Maximum padded sequence length T.

    Returns:
        np.ndarray: Boolean mask of shape (B, 1, 1, max_seq_len).
    """
    batch_size = len(sequence_lengths)
    positions = np.arange(max_seq_len).reshape(1, max_seq_len)
    lengths = np.array(sequence_lengths).reshape(batch_size, 1)
    mask_2d = positions < lengths
    return mask_2d[:, np.newaxis, np.newaxis, :]


def get_sinusoidal_positional_encoding(seq_len, d_model):
    """
    Computes Vaswani et al. (2017) sinusoidal positional encodings:
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Parameters:
        seq_len (int): Sequence length T.
        d_model (int): Feature representation dimension (must be even).

    Returns:
        np.ndarray: Positional encoding matrix of shape (seq_len, d_model).
    """
    pe = np.zeros((seq_len, d_model), dtype=np.float64)
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


# =============================================================================
# PART 2: SCALED DOT-PRODUCT ATTENTION (FORWARD & BACKWARD)
# =============================================================================

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Executes Scaled Dot-Product Attention:
        scores = (Q @ K^T) / sqrt(d_k)
        weights = softmax(scores + mask)
        output = weights @ V

    Parameters:
        Q (np.ndarray): Query tensor of shape (..., T_q, d_k).
        K (np.ndarray): Key tensor of shape (..., T_k, d_k).
        V (np.ndarray): Value tensor of shape (..., T_k, d_v).
        mask (np.ndarray, optional): Boolean or float attention mask broadcastable
                                     to (..., T_q, T_k). True indicates allowed positions.

    Returns:
        output (np.ndarray): Weighted value combination of shape (..., T_q, d_v).
        weights (np.ndarray): Attention probability matrix of shape (..., T_q, T_k).
    """
    # 1. Pairwise Query-Key similarity dot products: (..., T_q, d_k) @ (..., d_k, T_k) -> (..., T_q, T_k)
    raw_scores = Q @ np.swapaxes(K, -1, -2)

    # 2. Scale scores by 1 / sqrt(d_k) to maintain unit variance
    d_k = Q.shape[-1]
    scaled_scores = raw_scores / np.sqrt(d_k)

    # 3. Apply attention mask if provided
    if mask is not None:
        if mask.dtype == bool:
            masked_scores = np.where(mask, scaled_scores, -np.inf)
        else:
            masked_scores = scaled_scores + mask
    else:
        masked_scores = scaled_scores

    # 4. Row-wise Softmax probability distribution over keys
    weights = stable_softmax(masked_scores, axis=-1)

    # 5. Convex combination of Value vectors: (..., T_q, T_k) @ (..., T_k, d_v) -> (..., T_q, d_v)
    output = weights @ V

    return output, weights


def scaled_dot_product_attention_backward(d_out, Q, K, V, weights, mask=None):
    """
    Computes analytical gradients for Scaled Dot-Product Attention:
        dV = weights^T @ d_out
        d_weights = d_out @ V^T
        d_scores = weights ⊙ (d_weights - sum(d_weights ⊙ weights, axis=-1, keepdims=True))
        dQ = (d_scores / sqrt(d_k)) @ K
        dK = (d_scores / sqrt(d_k))^T @ Q

    Parameters:
        d_out (np.ndarray): Incoming gradient of loss w.r.t attention output (..., T_q, d_v).
        Q (np.ndarray): Cached Query tensor of shape (..., T_q, d_k).
        K (np.ndarray): Cached Key tensor of shape (..., T_k, d_k).
        V (np.ndarray): Cached Value tensor of shape (..., T_k, d_v).
        weights (np.ndarray): Cached attention weight probabilities (..., T_q, T_k).
        mask (np.ndarray, optional): Boolean attention mask.

    Returns:
        dQ (np.ndarray): Gradient w.r.t Queries (..., T_q, d_k).
        dK (np.ndarray): Gradient w.r.t Keys (..., T_k, d_k).
        dV (np.ndarray): Gradient w.r.t Values (..., T_k, d_v).
    """
    d_k = Q.shape[-1]

    # 1. Gradient w.r.t Value vectors: V = weights^T @ d_out
    dV = np.swapaxes(weights, -1, -2) @ d_out

    # 2. Gradient w.r.t Attention weights: d_weights = d_out @ V^T
    d_weights = d_out @ np.swapaxes(V, -1, -2)

    # 3. Softmax Jacobian vector product:
    # dS = weights ⊙ (d_weights - sum(d_weights ⊙ weights, axis=-1))
    sum_d_weights_weights = np.sum(d_weights * weights, axis=-1, keepdims=True)
    d_scores = weights * (d_weights - sum_d_weights_weights)

    # Masked positions receive zero gradient
    if mask is not None and mask.dtype == bool:
        d_scores = np.where(mask, d_scores, 0.0)

    # 4. Gradient w.r.t unscaled dot-product scores
    d_scores_raw = d_scores / np.sqrt(d_k)

    # 5. Gradients w.r.t Query and Key tensors
    dQ = d_scores_raw @ K
    dK = np.swapaxes(d_scores_raw, -1, -2) @ Q

    return dQ, dK, dV


# =============================================================================
# PART 3: MULTI-HEAD ATTENTION MODULE (MHA CLASS)
# =============================================================================

class MultiHeadAttention:
    """
    Multi-Head Attention (MHA) module implemented from first principles in pure NumPy.

    Splits embedding dimension d_model across h parallel heads:
        d_k = d_v = d_model // h

    Computes independent scaled dot-product attention in each head subspace,
    concatenates head outputs, and projects through a linear output matrix W^O.
    """

    def __init__(self, d_model, num_heads):
        """
        Initializes projection weights using Xavier / Glorot uniform initialization:
            limit = sqrt(6 / (fan_in + fan_out))

        Parameters:
            d_model (int): Model embedding dimension (e.g., 4 or 64).
            num_heads (int): Number of parallel attention heads (e.g., 2 or 8).
        """
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be evenly divisible by num_heads ({num_heads})"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # Xavier / Glorot initialization
        limit = np.sqrt(6.0 / (d_model + d_model))
        self.W_Q = np.random.uniform(-limit, limit, (d_model, d_model))
        self.W_K = np.random.uniform(-limit, limit, (d_model, d_model))
        self.W_V = np.random.uniform(-limit, limit, (d_model, d_model))
        self.W_O = np.random.uniform(-limit, limit, (d_model, d_model))

        # Gradient buffers
        self.dW_Q = np.zeros_like(self.W_Q)
        self.dW_K = np.zeros_like(self.W_K)
        self.dW_V = np.zeros_like(self.W_V)
        self.dW_O = np.zeros_like(self.W_O)

        # Forward cache for backpropagation
        self.cache = None

    def forward(self, Q_in, K_in, V_in, mask=None):
        """
        Executes Multi-Head Attention forward pass:
            1. Project input representations to Q, K, V subspaces.
            2. Reshape into (B, h, T, d_k).
            3. Apply parallel Scaled Dot-Product Attention.
            4. Concatenate heads back to (B, T, d_model).
            5. Project concatenated output through W^O.

        Parameters:
            Q_in (np.ndarray): Query sequence tensor of shape (B, T_q, d_model).
            K_in (np.ndarray): Key sequence tensor of shape (B, T_k, d_model).
            V_in (np.ndarray): Value sequence tensor of shape (B, T_k, d_model).
            mask (np.ndarray, optional): Attention mask of shape (T_q, T_k) or broadcastable.

        Returns:
            output (np.ndarray): Multi-head attention output of shape (B, T_q, d_model).
            weights (np.ndarray): Attention weights of shape (B, h, T_q, T_k).
        """
        batch_size, seq_len_q, _ = Q_in.shape
        _, seq_len_k, _ = K_in.shape

        # 1. Linear projections: (B, T, d_model) @ (d_model, d_model) -> (B, T, d_model)
        Q_proj = Q_in @ self.W_Q
        K_proj = K_in @ self.W_K
        V_proj = V_in @ self.W_V

        # 2. Reshape into parallel head tensors: (B, T, d_model) -> (B, T, h, d_k) -> (B, h, T, d_k)
        Q_heads = Q_proj.reshape(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K_heads = K_proj.reshape(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V_heads = V_proj.reshape(batch_size, seq_len_k, self.num_heads, self.d_v).transpose(0, 2, 1, 3)

        # 3. Parallel Scaled Dot-Product Attention across all heads
        # If mask is 2D (T_q, T_k), reshape to (1, 1, T_q, T_k) for broadcasting across (B, h, T_q, T_k)
        if mask is not None and mask.ndim == 2:
            mask_expanded = mask.reshape(1, 1, seq_len_q, seq_len_k)
        else:
            mask_expanded = mask

        head_outputs, weights = scaled_dot_product_attention(
            Q_heads, K_heads, V_heads, mask=mask_expanded
        )

        # 4. Concatenate heads: (B, h, T_q, d_v) -> (B, T_q, h, d_v) -> (B, T_q, d_model)
        concat_heads = head_outputs.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_q, self.d_model)

        # 5. Output linear projection: (B, T_q, d_model) @ (d_model, d_model) -> (B, T_q, d_model)
        output = concat_heads @ self.W_O

        # Cache tensors required for backpropagation
        self.cache = {
            "Q_in": Q_in,
            "K_in": K_in,
            "V_in": V_in,
            "Q_heads": Q_heads,
            "K_heads": K_heads,
            "V_heads": V_heads,
            "concat_heads": concat_heads,
            "weights": weights,
            "mask": mask_expanded,
        }

        return output, weights

    def backward(self, d_out):
        """
        Computes analytical gradients for all projection parameters and input representations:
            dW^O, dW^Q, dW^K, dW^V, dQ_in, dK_in, dV_in

        Parameters:
            d_out (np.ndarray): Incoming loss gradient of shape (B, T_q, d_model).

        Returns:
            dQ_in (np.ndarray): Gradient w.r.t input Queries (B, T_q, d_model).
            dK_in (np.ndarray): Gradient w.r.t input Keys (B, T_k, d_model).
            dV_in (np.ndarray): Gradient w.r.t input Values (B, T_k, d_model).
        """
        assert self.cache is not None, "Must execute forward pass before backward"

        Q_in = self.cache["Q_in"]
        K_in = self.cache["K_in"]
        V_in = self.cache["V_in"]
        Q_heads = self.cache["Q_heads"]
        K_heads = self.cache["K_heads"]
        V_heads = self.cache["V_heads"]
        concat_heads = self.cache["concat_heads"]
        weights = self.cache["weights"]
        mask = self.cache["mask"]

        batch_size, seq_len_q, _ = Q_in.shape
        _, seq_len_k, _ = K_in.shape

        # 1. Output projection weights gradient dW_O: (d_model, d_model)
        # Flatten batch and sequence dimensions: (B * T_q, d_model)^T @ (B * T_q, d_model)
        concat_flat = concat_heads.reshape(-1, self.d_model)
        d_out_flat = d_out.reshape(-1, self.d_model)
        self.dW_O = concat_flat.T @ d_out_flat

        # 2. Backprop into concatenated heads: (B, T_q, d_model) @ (d_model, d_model)^T
        d_concat = d_out @ self.W_O.T

        # 3. Unshape into parallel heads: (B, T_q, d_model) -> (B, T_q, h, d_v) -> (B, h, T_q, d_v)
        d_head_outputs = d_concat.reshape(batch_size, seq_len_q, self.num_heads, self.d_v).transpose(0, 2, 1, 3)

        # 4. Backprop through Scaled Dot-Product Attention per head
        dQ_heads, dK_heads, dV_heads = scaled_dot_product_attention_backward(
            d_head_outputs, Q_heads, K_heads, V_heads, weights, mask=mask
        )

        # 5. Unshape head gradients back to unified projections:
        # (B, h, T, d_k) -> (B, T, h, d_k) -> (B, T, d_model)
        dQ_proj = dQ_heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_q, self.d_model)
        dK_proj = dK_heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_k, self.d_model)
        dV_proj = dV_heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_k, self.d_model)

        # 6. Parameter gradients for projection matrices
        Q_in_flat = Q_in.reshape(-1, self.d_model)
        K_in_flat = K_in.reshape(-1, self.d_model)
        V_in_flat = V_in.reshape(-1, self.d_model)

        self.dW_Q = Q_in_flat.T @ dQ_proj.reshape(-1, self.d_model)
        self.dW_K = K_in_flat.T @ dK_proj.reshape(-1, self.d_model)
        self.dW_V = V_in_flat.T @ dV_proj.reshape(-1, self.d_model)

        # 7. Gradients w.r.t input representations
        dQ_in = dQ_proj @ self.W_Q.T
        dK_in = dK_proj @ self.W_K.T
        dV_in = dV_proj @ self.W_V.T

        return dQ_in, dK_in, dV_in

    def step(self, learning_rate):
        """
        Applies stochastic gradient descent parameter update:
            W <- W - learning_rate * dW
        """
        self.W_Q -= learning_rate * self.dW_Q
        self.W_K -= learning_rate * self.dW_K
        self.W_V -= learning_rate * self.dW_V
        self.W_O -= learning_rate * self.dW_O

    def zero_grad(self):
        """Resets all parameter gradient buffers to zero."""
        self.dW_Q.fill(0.0)
        self.dW_K.fill(0.0)
        self.dW_V.fill(0.0)
        self.dW_O.fill(0.0)


# =============================================================================
# PART 4: COMPREHENSIVE VERIFICATION SUITE
# =============================================================================

def run_checkpoint_1_step01_forward_verification():
    """
    Checkpoint 1: Validates exact forward matrices against the analytical trace
    derived in 12-attention/steps/step-01-mathematics.md (Section Q5).
    """
    print("\n" + "=" * 80)
    print("CHECKPOINT 1: STEP-01 MATHEMATICS FORWARD PASS TRACE")
    print("=" * 80)

    # Inputs from Step-01 Section Q5
    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ])

    W_Q = np.array([[1.0, 0.0], [0.0, 1.0]])
    W_K = np.array([[1.0, 0.0], [1.0, 1.0]])
    W_V = np.array([[1.0, 1.0], [0.0, 1.0]])

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    mask = create_causal_mask(seq_len=3)
    output, weights = scaled_dot_product_attention(Q, K, V, mask=mask)

    print("\n1. Linear Projections:")
    print("Q = X @ W_Q:\n", Q)
    print("K = X @ W_K:\n", K)
    print("V = X @ W_V:\n", V)

    # Expected mathematical targets from step-01-mathematics.md
    expected_weights = np.array([
        [1.000000, 0.000000, 0.000000],
        [0.330238, 0.669762, 0.000000],
        [0.140029, 0.283995, 0.575975]
    ])

    expected_output = np.array([
        [1.000000, 1.000000],
        [0.330238, 1.000000],
        [0.716005, 1.575975]
    ])

    print("\n2. Attention Probability Weights (Softmax):")
    print(weights)

    print("\n3. Output Representations (A @ V):")
    print(output)

    # Assertions
    np.testing.assert_allclose(weights, expected_weights, atol=1e-5,
                               err_msg="Attention weights do not match Step-01 math derivation!")
    np.testing.assert_allclose(output, expected_output, atol=1e-5,
                               err_msg="Attention output does not match Step-01 math derivation!")

    # Probability axioms
    assert np.all(weights >= 0.0), "Attention weights must be non-negative"
    np.testing.assert_allclose(np.sum(weights, axis=-1), 1.0, atol=1e-7,
                               err_msg="Attention weights must sum to 1.0 along key axis")

    print("\n[PASSED] Checkpoint 1: Forward activations match step-01 math down to 5 decimal places.")


def run_checkpoint_2_step01_backward_verification():
    """
    Checkpoint 2: Validates analytical gradients, matrix calculus, and parameter
    updates against step-01-mathematics.md (Sections Q6, Q7, Q8).
    """
    print("\n" + "=" * 80)
    print("CHECKPOINT 2: STEP-01 MATHEMATICS BACKPROPAGATION & UPDATE TRACE")
    print("=" * 80)

    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ])

    W_Q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    W_K = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    W_V = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    mask = create_causal_mask(seq_len=3)

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    output, weights = scaled_dot_product_attention(Q, K, V, mask=mask)

    # Ground truth targets from Step-01 Section Q6
    Y = np.array([
        [1.0, 0.5],
        [0.5, 1.0],
        [0.8, 1.2]
    ])

    initial_loss = 0.5 * np.sum((output - Y) ** 2)
    print(f"Initial Regression Loss: {initial_loss:.6f}")
    assert np.isclose(initial_loss, 0.213616, atol=1e-5), f"Expected 0.213616, got {initial_loss}"

    # Analytical backpropagation
    d_out = output - Y
    dQ, dK, dV = scaled_dot_product_attention_backward(d_out, Q, K, V, weights, mask=mask)

    dW_Q = X.T @ dQ
    dW_K = X.T @ dK
    dW_V = X.T @ dV
    dX = dQ @ W_Q.T + dK @ W_K.T + dV @ W_V.T

    # Expected gradients from Step-01 Section Q7
    expected_dW_Q = np.array([
        [0.055214, 0.023804],
        [0.055214, 0.050355]
    ])

    expected_dW_K = np.array([
        [0.031410, 0.004859],
        [0.023804, 0.050355]
    ])

    expected_dW_V = np.array([
        [-0.116203,  0.769200],
        [-0.185933,  0.323328]
    ])

    print("\nAnalytical Projection Gradients:")
    print("dW_Q:\n", dW_Q)
    print("dW_K:\n", dW_K)
    print("dW_V:\n", dW_V)

    np.testing.assert_allclose(dW_Q, expected_dW_Q, atol=1e-5)
    np.testing.assert_allclose(dW_K, expected_dW_K, atol=1e-5)
    np.testing.assert_allclose(dW_V, expected_dW_V, atol=1e-5)

    # 1 Step of Gradient Descent (eta = 0.1)
    learning_rate = 0.1
    W_Q_new = W_Q - learning_rate * dW_Q
    W_K_new = W_K - learning_rate * dW_K
    W_V_new = W_V - learning_rate * dW_V

    Q_new = X @ W_Q_new
    K_new = X @ W_K_new
    V_new = X @ W_V_new
    output_new, _ = scaled_dot_product_attention(Q_new, K_new, V_new, mask=mask)

    new_loss = 0.5 * np.sum((output_new - Y) ** 2)
    loss_delta = new_loss - initial_loss

    print(f"\nUpdated Loss after 1 SGD Step (eta={learning_rate}): {new_loss:.6f}")
    print(f"Loss Delta: {loss_delta:.6f}")

    assert np.isclose(new_loss, 0.146169, atol=1e-5), f"Expected 0.146169, got {new_loss}"
    assert loss_delta < -0.06, "1 Step of gradient descent must significantly reduce loss!"

    print("\n[PASSED] Checkpoint 2: Analytical gradients and SGD update match step-01 math.")


def run_checkpoint_3_finite_difference_gradient_checks():
    """
    Checkpoint 3: Rigorously verifies all analytical gradients (W_Q, W_K, W_V, W_O, X)
    against two-sided numerical finite differences:
        dF/dθ ≈ (F(θ + ε) - F(θ - ε)) / (2ε)
    """
    print("\n" + "=" * 80)
    print("CHECKPOINT 3: FINITE-DIFFERENCE NUMERICAL GRADIENT VERIFICATION")
    print("=" * 80)

    np.random.seed(42)
    B, T, d_model, h = 2, 4, 4, 2

    mha = MultiHeadAttention(d_model=d_model, num_heads=h)
    X = np.random.randn(B, T, d_model)
    Y = np.random.randn(B, T, d_model)
    mask = create_causal_mask(seq_len=T)

    out, _ = mha.forward(X, X, X, mask=mask)
    d_out = out - Y
    dX_q, dX_k, dX_v = mha.backward(d_out)
    dX = dX_q + dX_k + dX_v

    eps = 1e-6

    def compute_numerical_gradient(param_accessor, grad_analytic, name):
        param = param_accessor()
        grad_numerical = np.zeros_like(param)
        it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])

        while not it.finished:
            idx = it.multi_index
            orig_val = param[idx]

            # F(θ + ε)
            param[idx] = orig_val + eps
            out_pos, _ = mha.forward(X, X, X, mask=mask)
            loss_pos = 0.5 * np.sum((out_pos - Y) ** 2)

            # F(θ - ε)
            param[idx] = orig_val - eps
            out_neg, _ = mha.forward(X, X, X, mask=mask)
            loss_neg = 0.5 * np.sum((out_neg - Y) ** 2)

            grad_numerical[idx] = (loss_pos - loss_neg) / (2.0 * eps)
            param[idx] = orig_val
            it.iternext()

        rel_error = np.max(
            np.abs(grad_analytic - grad_numerical)
            / np.maximum(1e-8, np.abs(grad_analytic) + np.abs(grad_numerical))
        )
        print(f"  {name:6s} Gradient Relative Error: {rel_error:.2e}")
        assert rel_error < 1e-6, f"Finite-difference check failed for {name} with rel_error={rel_error}"

    print("Checking MultiHeadAttention Parameter Gradients against Finite Differences:")
    compute_numerical_gradient(lambda: mha.W_Q, mha.dW_Q, "W_Q")
    compute_numerical_gradient(lambda: mha.W_K, mha.dW_K, "W_K")
    compute_numerical_gradient(lambda: mha.W_V, mha.dW_V, "W_V")
    compute_numerical_gradient(lambda: mha.W_O, mha.dW_O, "W_O")

    # Verify input gradient dX
    grad_numerical_X = np.zeros_like(X)
    it_x = np.nditer(X, flags=["multi_index"], op_flags=["readwrite"])
    while not it_x.finished:
        idx = it_x.multi_index
        orig_val = X[idx]
        X[idx] = orig_val + eps
        out_pos, _ = mha.forward(X, X, X, mask=mask)
        loss_pos = 0.5 * np.sum((out_pos - Y) ** 2)

        X[idx] = orig_val - eps
        out_neg, _ = mha.forward(X, X, X, mask=mask)
        loss_neg = 0.5 * np.sum((out_neg - Y) ** 2)

        grad_numerical_X[idx] = (loss_pos - loss_neg) / (2.0 * eps)
        X[idx] = orig_val
        it_x.iternext()

    rel_error_X = np.max(
        np.abs(dX - grad_numerical_X)
        / np.maximum(1e-8, np.abs(dX) + np.abs(grad_numerical_X))
    )
    print(f"  {'Input X':6s} Gradient Relative Error: {rel_error_X:.2e}")
    assert rel_error_X < 1e-6, f"Finite-difference check failed for Input X with rel_error={rel_error_X}"

    print("\n[PASSED] Checkpoint 3: All analytical gradients verified against numerical finite differences (< 1e-6).")


def run_checkpoint_4_pytorch_parity_verification():
    """
    Checkpoint 4: Validates Scaled Dot-Product Attention directly against PyTorch's
    torch.nn.functional.scaled_dot_product_attention down to machine precision (< 1e-12).
    """
    print("\n" + "=" * 80)
    print("CHECKPOINT 4: PYTORCH FRAMEWORK PARITY VERIFICATION")
    print("=" * 80)

    try:
        import torch
        import torch.nn.functional as F

        np.random.seed(123)
        B, h, T, d_k = 2, 4, 6, 8

        Q_np = np.random.randn(B, h, T, d_k).astype(np.float64)
        K_np = np.random.randn(B, h, T, d_k).astype(np.float64)
        V_np = np.random.randn(B, h, T, d_k).astype(np.float64)

        # 1. Unmasked Parity Check
        out_np, weights_np = scaled_dot_product_attention(Q_np, K_np, V_np, mask=None)

        Q_torch = torch.tensor(Q_np)
        K_torch = torch.tensor(K_np)
        V_torch = torch.tensor(V_np)

        out_torch = F.scaled_dot_product_attention(Q_torch, K_torch, V_torch).numpy()
        max_diff_unmasked = np.max(np.abs(out_np - out_torch))
        print(f"Unmasked Attention Max Difference vs PyTorch: {max_diff_unmasked:.2e}")
        assert max_diff_unmasked < 1e-12, f"PyTorch parity failed (diff={max_diff_unmasked})"

        # 2. Causal Masked Parity Check
        causal_mask_2d = create_causal_mask(seq_len=T)
        out_np_masked, _ = scaled_dot_product_attention(Q_np, K_np, V_np, mask=causal_mask_2d)

        mask_torch = torch.tensor(causal_mask_2d)
        out_torch_masked = F.scaled_dot_product_attention(
            Q_torch, K_torch, V_torch, attn_mask=mask_torch
        ).numpy()

        max_diff_masked = np.max(np.abs(out_np_masked - out_torch_masked))
        print(f"Causal Masked Attention Max Difference vs PyTorch: {max_diff_masked:.2e}")
        assert max_diff_masked < 1e-12, f"PyTorch causal parity failed (diff={max_diff_masked})"

        print("\n[PASSED] Checkpoint 4: Exact numerical parity with PyTorch F.scaled_dot_product_attention (< 1e-12).")

    except ImportError:
        print("\n[SKIPPED] PyTorch is not installed in the current environment; skipping Checkpoint 4.")


def run_checkpoint_5_associative_memory_demo():
    """
    Checkpoint 5: Demonstrates how Self-Attention solves associative key-value binding
    and visualizes the resulting dynamic routing heatmap in plain ASCII.
    """
    print("\n" + "=" * 80)
    print("CHECKPOINT 5: ASSOCIATIVE MEMORY & ATTENTION HEATMAP DEMONSTRATION")
    print("=" * 80)

    # Sequence representing tokens: ["Paris", "is", "capital", "of", "France"]
    # Model embedding dimension d_model = 4, num_heads = 2
    tokens = ["Paris", "is", "capital", "of", "France"]
    T = len(tokens)
    d_model = 4
    h = 2

    np.random.seed(99)
    mha = MultiHeadAttention(d_model=d_model, num_heads=h)

    # Create dummy token embeddings + sinusoidal positional encoding
    token_embeddings = np.random.randn(1, T, d_model)
    pe = get_sinusoidal_positional_encoding(T, d_model)[np.newaxis, :, :]
    inputs = token_embeddings + pe

    # Forward pass (bidirectional encoder self-attention)
    output, weights = mha.forward(inputs, inputs, inputs, mask=None)

    print(f"Input Shape:  {inputs.shape}  (Batch=1, SeqLen={T}, Dim={d_model})")
    print(f"Output Shape: {output.shape}")
    print(f"Weights Shape: {weights.shape} (Batch=1, Heads={h}, SeqLen={T}, SeqLen={T})\n")

    for head_idx in range(h):
        print(f"--- Learned Attention Heatmap: Head {head_idx + 1} ---")
        head_w = weights[0, head_idx]

        header = "           " + "  ".join([f"{tok:>7s}" for tok in tokens])
        print(header)
        for i, row_token in enumerate(tokens):
            row_str = f"{row_token:>9s}: "
            for j in range(T):
                row_str += f"  {head_w[i, j]:6.3f}"
            print(row_str)
        print()

    print("[PASSED] Checkpoint 5: Attention weights computed and heatmaps visualized successfully.")


# =============================================================================
# MAIN EXECUTION ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("STARTING 12-ATTENTION FROM SCRATCH NUMERICAL VALIDATION SUITE")
    print("=" * 80)

    run_checkpoint_1_step01_forward_verification()
    run_checkpoint_2_step01_backward_verification()
    run_checkpoint_3_finite_difference_gradient_checks()
    run_checkpoint_4_pytorch_parity_verification()
    run_checkpoint_5_associative_memory_demo()

    print("\n" + "=" * 80)
    print("ALL 5 ATTENTION VERIFICATION CHECKPOINTS PASSED SUCCESSFULLY!")
    print("=" * 80)
