"""
11 — Sequence Prediction & Autoregressive Language Modeling From Scratch
========================================================================

Architecture Overview:
----------------------
This module implements a complete Autoregressive Neural Language Model from
first principles using only Python and pure NumPy. It establishes the mathematical
and algorithmic bedrock for all generative sequence prediction architectures—
from classical Recurrent Neural Networks (RNNs) to modern Large Language Models (LLMs).

The Autoregressive Objective:
-----------------------------
Under the probabilistic chain rule, the joint probability of any discrete text
sequence w = (w_1, w_2, ..., w_T) decomposes into an ordered product of conditional
next-token probabilities:
    P(w_1, w_2, ..., w_T) = ∏_{t=1}^T P(w_t | w_1, w_2, ..., w_{t-1})

To train this architecture, datasets are structured with a causal shift of +1:
    Input Sequence X:  [ w_0,   w_1,   ..., w_{T-1} ]  (Context tokens)
    Target Sequence Y: [ w_1,   w_2,   ..., w_T     ]  (Next-token ground truth)

At each timestep t:
1. Token Embedding:     e_t = E[w_{t-1}] ∈ R^{1 x D}
2. Recurrent Context:   h_t = tanh(e_t @ W_xh + h_{t-1} @ W_hh + b_h) ∈ R^{1 x H}
3. Vocabulary Readout:  o_t = h_t @ W_hy + b_y ∈ R^{1 x V}
4. Probability Dist:    y_hat_t = softmax(o_t) ∈ (0, 1)^{1 x V}
5. Causal Token Loss:   L_t = -ln(y_hat_{t, w_t})

Evaluation Metrics:
-------------------
1. Total Sequence Cross-Entropy:  L_total = sum_{t=1}^T L_t
2. Mean Token Loss:               L_bar = (1 / T) * L_total
3. Perplexity (PPL):              PPL = exp(L_bar)
   (Represents the effective branching factor / number of equally likely choices).

Tensor Dimension Conventions:
-----------------------------
- B: Batch size                       (e.g., B = 1 in verification)
- T: Sequential timesteps / tokens    (e.g., T = 3 in verification)
- V: Vocabulary size                  (e.g., V = 5)
- D: Embedding feature dimension      (e.g., D = 3)
- H: Hidden representation dimension  (e.g., H = 4)

Learnable Parameter Budget:
---------------------------
- Embedding Table:        E    (V, D) -> V*D parameters
- Input-to-Hidden:        W_xh (D, H) -> D*H parameters
- Recurrent Transition:   W_hh (H, H) -> H*H parameters
- Hidden Bias:            b_h  (1, H) -> H parameters
- Vocabulary Projection:  W_hy (H, V) -> H*V parameters
- Vocabulary Bias:        b_y  (1, V) -> V parameters
Total Learnable Parameters: V*D + D*H + H^2 + H + H*V + V
"""

import numpy as np


# =============================================================================
# PART 1: ACTIVATIONS AND VOCABULARY LINEAR PROJECTIONS
# =============================================================================

def stable_softmax(logits, temperature=1.0):
    """
    Computes numerically stable softmax probabilities across final vocabulary axis:
        p_k = exp((o_k - max(o)) / T) / sum_j exp((o_j - max(o)) / T)

    Parameters:
        logits (np.ndarray): Unnormalized prediction scores of shape (..., V).
        temperature (float): Positive scaling hyperparameter (T > 0).

    Returns:
        np.ndarray: Categorical probability distribution summing to 1.0.
    """
    scaled_logits = logits / max(temperature, 1e-8)
    max_per_sample = np.max(scaled_logits, axis=-1, keepdims=True)
    stabilized_exponentials = np.exp(scaled_logits - max_per_sample)
    partition_function = np.sum(stabilized_exponentials, axis=-1, keepdims=True)
    return stabilized_exponentials / partition_function


def dense_vocab_projection(hidden_state, output_weights, output_bias):
    """
    Linear projection mapping hidden representations to vocabulary logits:
        o_t = h_t @ W_{hy} + b_y

    Parameters:
        hidden_state (np.ndarray): Latent hidden representation of shape (B, H).
        output_weights (np.ndarray): Readout weight matrix of shape (H, V).
        output_bias (np.ndarray): Readout bias offset vector of shape (1, V).

    Returns:
        np.ndarray: Vocabulary logits of shape (B, V).
    """
    return hidden_state @ output_weights + output_bias


# =============================================================================
# PART 2: EMBEDDING LAYER & RECURRENT CELL FORWARD PASS (MICRO-FLOW)
# =============================================================================

def embedding_forward(token_indices, embedding_matrix):
    """
    Maps discrete integer token IDs into continuous dense coordinate vectors:
        e_t = E[w_t, :]

    Parameters:
        token_indices (np.ndarray): Integer token IDs of shape (B,) or (B, T).
        embedding_matrix (np.ndarray): Embedding table E of shape (V, D).

    Returns:
        np.ndarray: Dense embedding coordinates of shape (..., D).
    """
    return embedding_matrix[token_indices]


def recurrent_cell_forward(current_embedding, previous_hidden, parameters):
    """
    Executes a single recurrent context compression step at timestep t.

    Micro-Flow Numbered Steps:
        1. Input embedding affine projection: e_t @ W_xh
        2. Prior context affine projection:   h_{t-1} @ W_hh
        3. Fused affine pre-activation:       z_{h, t} = e_t @ W_xh + h_{t-1} @ W_hh + b_h
        4. Hyperbolic tangent bounding:       h_t = tanh(z_{h, t})
        5. Cache packaging for BPTT

    Parameters:
        current_embedding (np.ndarray): Input dense vector e_t of shape (B, D).
        previous_hidden (np.ndarray): Previous summary state h_{t-1} of shape (B, H).
        parameters (dict): Model parameter dictionary containing W_xh, W_hh, b_h.

    Returns:
        tuple: (current_hidden, step_cache)
            - current_hidden (np.ndarray): Activated hidden state h_t, shape (B, H).
            - step_cache (dict): Intermediate variables required for BPTT.
    """
    # -------------------------------------------------------------------------
    # Step 1 & 2 & 3: Fused Affine Recurrent Pre-Activation
    # -------------------------------------------------------------------------
    z_h_pre = (
        current_embedding @ parameters["W_xh"]
        + previous_hidden @ parameters["W_hh"]
        + parameters["b_h"]
    )  # Shape: (B, H)

    # -------------------------------------------------------------------------
    # Step 4: Non-Linear Hyperbolic Tangent Bounding
    # -------------------------------------------------------------------------
    current_hidden = np.tanh(z_h_pre)  # Shape: (B, H)

    # -------------------------------------------------------------------------
    # Step 5: Cache Packaging for BPTT
    # -------------------------------------------------------------------------
    step_cache = {
        "current_embedding": current_embedding,
        "previous_hidden": previous_hidden,
        "z_h_pre": z_h_pre,
        "current_hidden": current_hidden,
    }

    return current_hidden, step_cache


# =============================================================================
# PART 3: UNROLLED SEQUENCE FORWARD PASS (MESO-FLOW)
# =============================================================================

def sequence_model_forward(token_sequence, initial_hidden, parameters):
    """
    Unrolls the autoregressive sequence forward graph across timesteps t = 0..T-1.

    Parameters:
        token_sequence (np.ndarray): Integer token IDs of shape (B, T).
        initial_hidden (np.ndarray): Starting hidden state h_0 of shape (B, H).
        parameters (dict): Complete model parameter dictionary (E, W_xh, W_hh, b_h, W_hy, b_y).

    Returns:
        dict: Trajectories, output probabilities, and intermediate step caches:
            - all_embeddings: (B, T, D)
            - all_hidden_states: (B, T, H)
            - all_logits: (B, T, V)
            - all_output_probabilities: (B, T, V)
            - all_cell_caches: list of length T per-step caches
            - initial_hidden: h_0 of shape (B, H)
    """
    batch_size, sequence_length = token_sequence.shape
    embed_dim = parameters["E"].shape[-1]
    hidden_dim = initial_hidden.shape[-1]
    vocab_size = parameters["b_y"].shape[-1]

    all_embeddings = np.zeros((batch_size, sequence_length, embed_dim), dtype=np.float64)
    all_hidden_states = np.zeros((batch_size, sequence_length, hidden_dim), dtype=np.float64)
    all_logits = np.zeros((batch_size, sequence_length, vocab_size), dtype=np.float64)
    all_output_probabilities = np.zeros((batch_size, sequence_length, vocab_size), dtype=np.float64)
    all_cell_caches = []

    current_hidden = initial_hidden.copy()

    # Temporal unrolling loop across discrete timesteps t = 0, 1, ..., T-1
    for timestep in range(sequence_length):
        current_token_ids = token_sequence[:, timestep]  # Shape: (B,)
        current_embed = embedding_forward(current_token_ids, parameters["E"])  # Shape: (B, D)

        current_hidden, step_cache = recurrent_cell_forward(
            current_embedding=current_embed,
            previous_hidden=current_hidden,
            parameters=parameters,
        )

        step_logits = dense_vocab_projection(
            hidden_state=current_hidden,
            output_weights=parameters["W_hy"],
            output_bias=parameters["b_y"],
        )  # Shape: (B, V)

        step_probabilities = stable_softmax(step_logits)  # Shape: (B, V)

        all_embeddings[:, timestep, :] = current_embed
        all_hidden_states[:, timestep, :] = current_hidden
        all_logits[:, timestep, :] = step_logits
        all_output_probabilities[:, timestep, :] = step_probabilities
        all_cell_caches.append(step_cache)

    return {
        "all_embeddings": all_embeddings,
        "all_hidden_states": all_hidden_states,
        "all_logits": all_logits,
        "all_output_probabilities": all_output_probabilities,
        "all_cell_caches": all_cell_caches,
        "initial_hidden": initial_hidden,
    }


# =============================================================================
# PART 4: LOSS COMPUTATION AND PERPLEXITY EVALUATION
# =============================================================================

def sequence_cross_entropy(output_probabilities, target_tokens, epsilon=1e-15):
    """
    Computes Causal Negative Log-Likelihood (NLL) Loss and Perplexity across sequence:
        L_t = -ln(y_hat_{t, y_t^*} + eps)
        L_total = sum_{t=1}^T L_t
        L_bar = (1 / T) * L_total
        PPL = exp(L_bar)

    Parameters:
        output_probabilities (np.ndarray): Predicted class probabilities of shape (B, T, V).
        target_tokens (np.ndarray): Integer target IDs of shape (B, T).
        epsilon (float): Floor to prevent ln(0).

    Returns:
        tuple: (total_loss, average_loss, step_losses, perplexity)
            - total_loss (float): Sum of cross-entropy losses over all timesteps.
            - average_loss (float): Mean loss per token.
            - step_losses (list of float): Per-timestep scalar losses [L_1, ..., L_T].
            - perplexity (float): Geometric mean reciprocal probability exp(average_loss).
    """
    batch_size, sequence_length, _ = output_probabilities.shape
    step_losses = []

    for t in range(sequence_length):
        step_probs = output_probabilities[:, t, :]  # (B, V)
        target_ids = target_tokens[:, t]           # (B,)

        # Extract predicted probability assigned to ground-truth token
        target_probs = step_probs[np.arange(batch_size), target_ids]
        target_probs_clipped = np.clip(target_probs, epsilon, 1.0 - epsilon)

        loss_t = float(np.mean(-np.log(target_probs_clipped)))
        step_losses.append(loss_t)

    total_loss = float(np.sum(step_losses))
    average_loss = total_loss / sequence_length
    perplexity = float(np.exp(average_loss))

    return total_loss, average_loss, step_losses, perplexity


def output_score_gradients(output_probabilities, target_tokens):
    """
    Analytical gradient of the batch-mean cross-entropy with respect to
    unnormalized vocabulary logits o_t:
        delta_{logit, t} = dL_t / do_t
                           = (y_hat_t - 1_{y_t^*}) / B  ∈ R^{B x V}

    `sequence_cross_entropy` computes the mean loss over B samples at each
    timestep. Dividing by B here makes BPTT differentiate that reported loss,
    instead of a batch-summed loss.

    Parameters:
        output_probabilities (np.ndarray): Probabilities of shape (B, T, V).
        target_tokens (np.ndarray): Ground-truth token IDs of shape (B, T).

    Returns:
        np.ndarray: Logit error tensor of shape (B, T, V).
    """
    batch_size, sequence_length, vocab_size = output_probabilities.shape
    logit_errors = output_probabilities.copy()

    for t in range(sequence_length):
        target_ids = target_tokens[:, t]
        logit_errors[np.arange(batch_size), t, target_ids] -= 1.0

    return logit_errors / batch_size


# =============================================================================
# PART 5: BACKPROPAGATION THROUGH TIME (BPTT) WITH SPARSE EMBEDDING UPDATES
# =============================================================================

def sequence_model_backward(token_sequence, target_tokens, forward_results, parameters):
    """
    Executes Backpropagation Through Time (BPTT) with sparse embedding table accumulation.

    Error propagates backward in reverse chronological order (t = T-1 down to 0).
    At each step:
        1. delta_{logit, t} = y_hat_t - 1_{y_t^*}
        2. dW_hy += h_t.T @ delta_{logit, t}, db_y += sum(delta_{logit, t})
        3. delta_{h_t} = delta_{logit, t} @ W_{hy}.T + delta_{h_t, recurrent}
        4. delta_{z_h, t} = delta_{h_t} ⊙ (1 - h_t^2)
        5. dW_xh += e_t.T @ delta_{z_h, t}, dW_hh += h_{t-1}.T @ delta_{z_h, t}, db_h += sum(delta_{z_h, t})
        6. delta_{e_t} = delta_{z_h, t} @ W_xh.T
        7. np.add.at(dE, token_sequence[:, t], delta_{e_t})
        8. delta_{h_{t-1}, recurrent} = delta_{z_h, t} @ W_hh.T

    Parameters:
        token_sequence (np.ndarray): Input token IDs of shape (B, T).
        target_tokens (np.ndarray): Target token IDs of shape (B, T).
        forward_results (dict): Output from sequence_model_forward.
        parameters (dict): Active parameter dictionary.

    Returns:
        dict: Parameter gradients for E, W_xh, W_hh, b_h, W_hy, b_y.
    """
    _, sequence_length = token_sequence.shape
    all_output_probs = forward_results["all_output_probabilities"]
    all_cell_caches = forward_results["all_cell_caches"]

    logit_errors = output_score_gradients(all_output_probs, target_tokens)

    # Initialize parameter gradient accumulators
    param_gradients = {
        "E": np.zeros_like(parameters["E"]),
        "W_xh": np.zeros_like(parameters["W_xh"]),
        "W_hh": np.zeros_like(parameters["W_hh"]),
        "b_h": np.zeros_like(parameters["b_h"]),
        "W_hy": np.zeros_like(parameters["W_hy"]),
        "b_y": np.zeros_like(parameters["b_y"]),
    }

    # Recurrent error carried from future step (zero at terminal step T-1)
    future_hidden_error = np.zeros_like(forward_results["initial_hidden"])

    # Reverse temporal unrolling loop: t = T-1, T-2, ..., 0
    for timestep in reversed(range(sequence_length)):
        step_logit_error = logit_errors[:, timestep, :]  # Shape: (B, V)
        step_cache = all_cell_caches[timestep]

        e_t = step_cache["current_embedding"]
        h_prev = step_cache["previous_hidden"]
        h_t = step_cache["current_hidden"]
        token_ids_at_step = token_sequence[:, timestep]

        # 1. Output Vocabulary Projection Gradients
        param_gradients["W_hy"] += h_t.T @ step_logit_error
        param_gradients["b_y"] += np.sum(step_logit_error, axis=0, keepdims=True)

        # 2. Total Hidden State Error: Immediate Loss + Future Recurrent Influx
        direct_hidden_error = step_logit_error @ parameters["W_hy"].T
        total_hidden_error = direct_hidden_error + future_hidden_error  # (B, H)

        # 3. Pre-Activation Gradient through Tanh Activation
        delta_z_h = total_hidden_error * (1.0 - h_t**2)  # (B, H)

        # 4. Recurrent Weight Gradients
        param_gradients["W_xh"] += e_t.T @ delta_z_h
        param_gradients["W_hh"] += h_prev.T @ delta_z_h
        param_gradients["b_h"] += np.sum(delta_z_h, axis=0, keepdims=True)

        # 5. Embedding Vector Error & Sparse Table Accumulation
        delta_e_t = delta_z_h @ parameters["W_xh"].T  # (B, D)
        np.add.at(param_gradients["E"], token_ids_at_step, delta_e_t)

        # 6. Recurrent Error Propagating to Prior Timestep h_{t-1}
        future_hidden_error = delta_z_h @ parameters["W_hh"].T

    return param_gradients


# =============================================================================
# PART 6: AUTOREGRESSIVE DECODING & SAMPLING STRATEGIES
# =============================================================================

def sample_next_token(logits, method="greedy", temperature=1.0, top_k=None, top_p=None):
    """
    Decodes the next token from raw vocabulary prediction logits using standard sampling algorithms.

    Methods:
        - "greedy": Deterministic argmax selection.
        - "temperature": Scaled Softmax probability distribution sampling.
        - "top_k": Truncates candidates strictly to top K logits before sampling.
        - "top_p": Nucleus sampling filtering smallest set with cumulative probability >= p.

    Parameters:
        logits (np.ndarray): Unnormalized prediction scores of shape (V,) or (1, V).
        method (str): "greedy", "temperature", "top_k", or "top_p".
        temperature (float): Temperature scaling parameter (T > 0).
        top_k (int, optional): Top-K cutoff rank.
        top_p (float, optional): Nucleus cumulative probability mass threshold (0 < p <= 1).

    Returns:
        tuple: (selected_token_id, probability_distribution)
            - selected_token_id (int): Sampled integer token ID.
            - probability_distribution (np.ndarray): Evaluated probabilities over vocabulary.
    """
    flat_logits = logits.flatten().copy()
    vocab_size = flat_logits.shape[0]

    if method == "greedy":
        selected_token = int(np.argmax(flat_logits))
        probs = stable_softmax(flat_logits)
        return selected_token, probs

    # Apply Temperature Scaling
    scaled_logits = flat_logits / max(temperature, 1e-8)

    # Top-K Filtering
    if method == "top_k" and top_k is not None:
        k = min(max(1, top_k), vocab_size)
        top_k_indices = np.argpartition(scaled_logits, -k)[-k:]
        mask = np.full(vocab_size, -np.inf)
        mask[top_k_indices] = scaled_logits[top_k_indices]
        scaled_logits = mask

    # Top-P (Nucleus) Filtering
    elif method == "top_p" and top_p is not None:
        p = float(np.clip(top_p, 1e-5, 1.0))
        sorted_indices = np.argsort(scaled_logits)[::-1]
        sorted_logits = scaled_logits[sorted_indices]
        sorted_probs = stable_softmax(sorted_logits)
        cumulative_probs = np.cumsum(sorted_probs)

        # Identify cutoff index
        cutoff_mask = cumulative_probs >= p
        if np.any(cutoff_mask):
            cutoff_index = int(np.where(cutoff_mask)[0][0])
            valid_indices = sorted_indices[: cutoff_index + 1]
        else:
            valid_indices = sorted_indices

        mask = np.full(vocab_size, -np.inf)
        mask[valid_indices] = scaled_logits[valid_indices]
        scaled_logits = mask

    probs = stable_softmax(scaled_logits)
    # Re-normalize for numerical safety in random choice
    probs = probs / np.sum(probs)
    selected_token = int(np.random.choice(vocab_size, p=probs))

    return selected_token, probs


def generate_sequence(
    prompt_tokens,
    max_new_tokens,
    parameters,
    initial_hidden=None,
    method="greedy",
    temperature=1.0,
    top_k=None,
    top_p=None,
    eos_token_id=None,
):
    """
    Generates tokens autoregressively in free-running autonomous inference mode.

    Parameters:
        prompt_tokens (list of int): Prefix prompt integer IDs.
        max_new_tokens (int): Maximum number of new tokens to synthesize.
        parameters (dict): Model parameter dictionary.
        initial_hidden (np.ndarray, optional): Starting hidden state.
        method (str): Sampling method ("greedy", "temperature", "top_k", "top_p").
        temperature (float): Sampling temperature.
        top_k (int, optional): Top-K parameter.
        top_p (float, optional): Top-P nucleus parameter.
        eos_token_id (int, optional): Token ID indicating End of Sequence.

    Returns:
        list of int: Complete sequence of tokens (prompt + generated).
    """
    hidden_dim = parameters["b_h"].shape[-1]
    if initial_hidden is None:
        current_hidden = np.zeros((1, hidden_dim), dtype=np.float64)
    else:
        current_hidden = initial_hidden.copy()

    generated_sequence_tokens = list(prompt_tokens)

    # 1. Warm up hidden state by feeding prefix prompt
    for token_id in prompt_tokens:
        embed = embedding_forward(np.array([token_id]), parameters["E"])
        current_hidden, _ = recurrent_cell_forward(embed, current_hidden, parameters)

    # 2. Free-running autonomous generation loop
    for _ in range(max_new_tokens):
        # Project current context to vocabulary logits
        logits = dense_vocab_projection(
            hidden_state=current_hidden,
            output_weights=parameters["W_hy"],
            output_bias=parameters["b_y"],
        )

        next_token, _ = sample_next_token(
            logits,
            method=method,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        generated_sequence_tokens.append(next_token)

        if eos_token_id is not None and next_token == eos_token_id:
            break

        # Feed generated token back into model for next step
        embed = embedding_forward(np.array([next_token]), parameters["E"])
        current_hidden, _ = recurrent_cell_forward(embed, current_hidden, parameters)

    return generated_sequence_tokens


# =============================================================================
# PART 7: TRAINING AND OPTIMIZATION UTILITIES
# =============================================================================

def clip_gradients_by_norm(parameter_gradients, max_norm=5.0):
    """
    Clips parameter gradient dictionary by global L2 Frobenius norm:
        scale = min(1.0, max_norm / (||g|| + eps))

    Parameters:
        parameter_gradients (dict): Mapping from parameter string to gradient array.
        max_norm (float): Maximum permissible global L2 norm threshold.

    Returns:
        tuple: (clipped_gradients, global_norm)
    """
    squared_norm_sum = sum(np.sum(g**2) for g in parameter_gradients.values())
    global_norm = float(np.sqrt(squared_norm_sum))

    if global_norm > max_norm:
        rescale_factor = max_norm / (global_norm + 1e-12)
        clipped_gradients = {k: v * rescale_factor for k, v in parameter_gradients.items()}
        return clipped_gradients, global_norm

    return {k: v.copy() for k, v in parameter_gradients.items()}, global_norm


def gradient_descent_update(parameters, gradients, learning_rate):
    """
    Standard first-order Stochastic Gradient Descent (SGD) parameter update:
        theta = theta - lr * dL/dtheta

    Parameters:
        parameters (dict): Model parameter dictionary.
        gradients (dict): Parameter gradient dictionary.
        learning_rate (float): Optimization step size.

    Returns:
        dict: Updated parameter dictionary.
    """
    return {k: parameters[k] - learning_rate * gradients[k] for k in parameters}


def train_sequence_model(
    token_sequences,
    target_sequences,
    initial_parameters,
    initial_hidden,
    epochs=100,
    learning_rate=0.1,
    verbose=False,
):
    """
    Standard training optimization loop for autoregressive sequence prediction.

    Parameters:
        token_sequences (np.ndarray): Input token IDs of shape (B, T).
        target_sequences (np.ndarray): Target token IDs of shape (B, T).
        initial_parameters (dict): Starting parameter dictionary.
        initial_hidden (np.ndarray): Starting hidden state of shape (B, H).
        epochs (int): Number of training epochs.
        learning_rate (float): Optimization step size.
        verbose (bool): If True, logs loss and perplexity.

    Returns:
        tuple: (trained_parameters, loss_history, perplexity_history)
    """
    params = {k: v.copy() for k, v in initial_parameters.items()}
    loss_history = []
    perplexity_history = []

    for ep in range(epochs):
        forward_results = sequence_model_forward(token_sequences, initial_hidden, params)
        total_loss, avg_loss, _, ppl = sequence_cross_entropy(
            forward_results["all_output_probabilities"], target_sequences
        )
        loss_history.append(total_loss)
        perplexity_history.append(ppl)

        grads = sequence_model_backward(
            token_sequences, target_sequences, forward_results, params
        )
        clipped_grads, _ = clip_gradients_by_norm(grads, max_norm=5.0)
        params = gradient_descent_update(params, clipped_grads, learning_rate)

        if verbose and ((ep + 1) % 20 == 0 or ep == 0):
            print(f"Epoch {ep+1:3d}/{epochs} | Loss: {total_loss:.4f} | Perplexity: {ppl:.2f}")

    return params, loss_history, perplexity_history


# =============================================================================
# PART 8: FINITE-DIFFERENCE NUMERICAL GRADIENT VERIFICATION
# =============================================================================

def compute_numerical_gradient(
    token_sequence,
    target_tokens,
    initial_hidden,
    parameters,
    param_name,
    row,
    col,
    epsilon=1e-5,
):
    """
    Computes symmetric finite-difference quotient for coordinate θ[row, col]:
        dL/dθ ≈ [ L(θ + ε) - L(θ - ε) ] / (2 * ε)

    Parameters:
        token_sequence (np.ndarray): Shape (B, T).
        target_tokens (np.ndarray): Shape (B, T).
        initial_hidden (np.ndarray): Shape (B, H).
        parameters (dict): Model parameter dictionary.
        param_name (str): Key of tensor to perturb (e.g. "E", "W_xh", "b_y").
        row (int): Row index in target parameter tensor.
        col (int): Column index in target parameter tensor.
        epsilon (float): Symmetric perturbation offset.

    Returns:
        float: Numerically estimated gradient.
    """
    params_plus = {k: v.copy() for k, v in parameters.items()}
    params_minus = {k: v.copy() for k, v in parameters.items()}

    params_plus[param_name][row, col] += epsilon
    params_minus[param_name][row, col] -= epsilon

    fwd_plus = sequence_model_forward(token_sequence, initial_hidden, params_plus)
    loss_plus, _, _, _ = sequence_cross_entropy(
        fwd_plus["all_output_probabilities"], target_tokens
    )

    fwd_minus = sequence_model_forward(token_sequence, initial_hidden, params_minus)
    loss_minus, _, _, _ = sequence_cross_entropy(
        fwd_minus["all_output_probabilities"], target_tokens
    )

    return (loss_plus - loss_minus) / (2.0 * epsilon)


# =============================================================================
# PART 9: EXECUTION, NUMERICAL ASSERTIONS, AND PEDAGOGICAL DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("11 — AUTOREGRESSIVE SEQUENCE PREDICTION & LANGUAGE MODELING FROM SCRATCH")
    print("Demonstrating First-Principles Mathematics, Embedding Calculus, & Sampling")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # CHECKPOINT 1: Exact Numerical Walkthrough (step-01-mathematics.md)
    # -------------------------------------------------------------------------
    print("\n[CHECKPOINT 1] Validating Exact Numerical Walkthrough (step-01-mathematics.md)...")

    # Dimensions: V = 5, D = 3, H = 4, T = 3, B = 1
    # Vocabulary: 0: <BOS>, 1: crypto, 2: blocks, 3: verify, 4: <EOS>
    vocab = {0: "<BOS>", 1: "crypto", 2: "blocks", 3: "verify", 4: "<EOS>"}
    inv_vocab = {v: k for k, v in vocab.items()}

    # Input Sequence X: [0, 1, 2] ("<BOS> crypto blocks")
    # Target Sequence Y: [1, 2, 3] ("crypto blocks verify")
    X_toy = np.array([[0, 1, 2]], dtype=np.int64)  # Shape: (1, 3)
    Y_toy = np.array([[1, 2, 3]], dtype=np.int64)  # Shape: (1, 3)

    h_0 = np.zeros((1, 4), dtype=np.float64)

    # Parameters matching step-01-mathematics.md Q9
    toy_parameters = {
        "E": np.array([
            [ 0.2, -0.1,  0.3],   # 0: <BOS>
            [-0.3,  0.4,  0.1],   # 1: crypto
            [ 0.1, -0.2,  0.4],   # 2: blocks
            [ 0.3,  0.1, -0.2],   # 3: verify
            [-0.1,  0.2, -0.3],   # 4: <EOS>
        ], dtype=np.float64),
        "W_xh": np.array([
            [ 0.2, -0.3,  0.1,  0.4],
            [-0.1,  0.2, -0.4,  0.3],
            [ 0.3,  0.1,  0.2, -0.2],
        ], dtype=np.float64),
        "W_hh": np.array([
            [ 0.1, -0.2,  0.3,  0.1],
            [-0.1,  0.4,  0.1, -0.3],
            [ 0.2, -0.1,  0.2,  0.4],
            [ 0.3,  0.1, -0.2,  0.1],
        ], dtype=np.float64),
        "b_h": np.zeros((1, 4), dtype=np.float64),
        "W_hy": np.array([
            [ 0.3, -0.2,  0.4, -0.1,  0.2],
            [-0.2,  0.5, -0.1,  0.3, -0.4],
            [ 0.1, -0.3,  0.2,  0.4, -0.1],
            [ 0.4,  0.1, -0.2, -0.3,  0.5],
        ], dtype=np.float64),
        "b_y": np.zeros((1, 5), dtype=np.float64),
    }

    # Execute Forward Pass
    forward_results = sequence_model_forward(X_toy, h_0, toy_parameters)
    total_seq_loss, avg_loss, step_losses, ppl = sequence_cross_entropy(
        forward_results["all_output_probabilities"], Y_toy
    )

    print(f"  Step 1 Loss L_1: {step_losses[0]:.6f} (Expected: 1.722698)")
    print(f"  Step 2 Loss L_2: {step_losses[1]:.6f} (Expected: 1.665319)")
    print(f"  Step 3 Loss L_3: {step_losses[2]:.6f} (Expected: 1.515687)")
    print(f"  Total Sequence Loss: {total_seq_loss:.6f} (Expected: 4.903704)")
    print(f"  Mean Token Loss:     {avg_loss:.6f} (Expected: 1.634568)")
    print(f"  Perplexity (PPL):    {ppl:.6f} (Expected: 5.127242)")

    assert np.isclose(step_losses[0], 1.722698, atol=1e-4), "L_1 mismatch!"
    assert np.isclose(step_losses[1], 1.665319, atol=1e-4), "L_2 mismatch!"
    assert np.isclose(step_losses[2], 1.515687, atol=1e-4), "L_3 mismatch!"
    assert np.isclose(total_seq_loss, 4.903704, atol=1e-4), "Total Loss mismatch!"
    assert np.isclose(avg_loss, 1.634568, atol=1e-4), "Mean Loss mismatch!"
    assert np.isclose(ppl, 5.127242, atol=1e-4), "Perplexity mismatch!"
    print("  [SUCCESS] All forward losses and perplexity match Step-01 Mathematics!")

    # Execute Backward Pass (BPTT)
    analytical_grads = sequence_model_backward(X_toy, Y_toy, forward_results, toy_parameters)

    # Expected values from step-01-mathematics.md Q11
    expected_dE = np.array([
        [ 0.298253, -0.382551,  0.156063],
        [ 0.163130,  0.129872, -0.182303],
        [ 0.240485,  0.158833, -0.097656],
        [ 0.000000,  0.000000,  0.000000],
        [ 0.000000,  0.000000,  0.000000],
    ])
    expected_dW_hy = np.array([
        [ 0.049887, -0.095186,  0.085014, -0.086995,  0.047279],
        [ 0.020580,  0.074711, -0.098078, -0.017734,  0.020520],
        [ 0.030749, -0.096083,  0.143055, -0.105886,  0.028164],
        [-0.024828, -0.011604, -0.085786,  0.144733, -0.022515],
    ])
    expected_db_y = np.array([[0.596562, -0.414853, -0.376344, -0.384705, 0.579340]])

    expected_dW_xh = np.array([
        [ 0.177364, -0.116455,  0.066998, -0.105873],
        [-0.189430,  0.064909,  0.014745,  0.108035],
        [ 0.171709, -0.319151,  0.023507,  0.173863],
    ])
    expected_dW_hh = np.array([
        [-0.044364, -0.010897,  0.010027,  0.050071],
        [ 0.039606, -0.025682, -0.037652,  0.021334],
        [-0.055880,  0.012883,  0.034197,  0.013446],
        [ 0.014919, -0.013882, -0.017593,  0.015883],
    ])
    expected_db_h = np.array([[0.318333, -1.064314, 0.185957, 0.750779]])

    assert np.allclose(analytical_grads["E"], expected_dE, atol=1e-4), "dE mismatch!"
    assert np.allclose(analytical_grads["W_hy"], expected_dW_hy, atol=1e-4), "dW_hy mismatch!"
    assert np.allclose(analytical_grads["b_y"], expected_db_y, atol=1e-4), "db_y mismatch!"
    assert np.allclose(analytical_grads["W_xh"], expected_dW_xh, atol=1e-4), "dW_xh mismatch!"
    assert np.allclose(analytical_grads["W_hh"], expected_dW_hh, atol=1e-4), "dW_hh mismatch!"
    assert np.allclose(analytical_grads["b_h"], expected_db_h, atol=1e-4), "db_h mismatch!"
    print("  [SUCCESS] Analytical BPTT gradients (including sparse Embedding updates) match theoretical derivations!")

    # -------------------------------------------------------------------------
    # CHECKPOINT 2: Finite-Difference Gradient Checking
    # -------------------------------------------------------------------------
    print("\n[CHECKPOINT 2] Running Finite-Difference Gradient Checks...")
    param_checks = [
        ("E",    0, 1, "E[0, 1] (<BOS> Embedding)"),
        ("E",    1, 0, "E[1, 0] (crypto Embedding)"),
        ("E",    2, 2, "E[2, 2] (blocks Embedding)"),
        ("W_xh", 0, 2, "W_xh[0, 2] (Input-to-Hidden)"),
        ("W_hh", 1, 1, "W_hh[1, 1] (Recurrent Transition)"),
        ("b_h",  0, 1, "b_h[0, 1] (Hidden Bias)"),
        ("W_hy", 0, 0, "W_hy[0, 0] (Readout Weight)"),
        ("b_y",  0, 1, "b_y[0, 1] (Vocabulary Bias)"),
    ]

    for param_name, r, c, label in param_checks:
        anal_val = analytical_grads[param_name][r, c]
        num_val = compute_numerical_gradient(
            X_toy, Y_toy, h_0, toy_parameters, param_name, r, c
        )
        rel_error = abs(anal_val - num_val) / (abs(anal_val) + abs(num_val) + 1e-12)
        print(f"  Checking {label:<32} | Analytical: {anal_val:+.7f} | Numerical: {num_val:+.7f} | Rel Err: {rel_error:.2e}")
        assert rel_error < 1e-4, f"Gradient check failed for {label}!"

    print("  [SUCCESS] All sequence prediction gradients verified against symmetric numerical differentiation!")

    # The hand derivation above uses B = 1. Duplicate it into B = 2 to verify
    # that BPTT preserves the batch-mean loss scale for mini-batches as well.
    X_toy_batched = np.repeat(X_toy, repeats=2, axis=0)
    Y_toy_batched = np.repeat(Y_toy, repeats=2, axis=0)
    h_0_batched = np.zeros((2, 4), dtype=np.float64)
    batched_forward_results = sequence_model_forward(
        X_toy_batched, h_0_batched, toy_parameters
    )
    batched_analytical_grads = sequence_model_backward(
        X_toy_batched, Y_toy_batched, batched_forward_results, toy_parameters
    )
    batched_numerical_gradient = compute_numerical_gradient(
        X_toy_batched, Y_toy_batched, h_0_batched, toy_parameters, "W_xh", 0, 2
    )
    assert np.allclose(
        batched_analytical_grads["W_xh"], analytical_grads["W_xh"], atol=1e-10
    ), "Batch-mean gradients should equal the single-example gradient for duplicated data!"
    assert np.isclose(
        batched_analytical_grads["W_xh"][0, 2], batched_numerical_gradient, atol=1e-6
    ), "Batched finite-difference gradient mismatch!"
    print("  [SUCCESS] Batch-mean BPTT scaling verified for B = 2!")

    # -------------------------------------------------------------------------
    # CHECKPOINT 3: Decoding & Sampling Strategies on Step 3 Logits
    # -------------------------------------------------------------------------
    print("\n[CHECKPOINT 3] Validating Decoding & Sampling Algorithms on Step 3 Logits...")
    step_3_logits = forward_results["all_logits"][0, 2, :]  # Shape: (5,)
    print(f"  Step 3 Logits o_3: {np.round(step_3_logits, 6)}")

    # 1. Greedy Search (argmax)
    greedy_token, _ = sample_next_token(step_3_logits, method="greedy")
    print(f"  Greedy Token: {greedy_token} ({vocab[greedy_token]}) [Expected: 2 ('blocks')]")
    assert greedy_token == 2, "Greedy decoding mismatch!"

    # 2. Temperature Scaling
    _, p_cold = sample_next_token(step_3_logits, method="temperature", temperature=0.5)
    _, p_std  = sample_next_token(step_3_logits, method="temperature", temperature=1.0)
    _, p_hot  = sample_next_token(step_3_logits, method="temperature", temperature=2.0)
    print(f"  Cold Temp (T=0.5) Probs: {np.round(p_cold, 4)} (Sharpened around top tokens)")
    print(f"  Std  Temp (T=1.0) Probs: {np.round(p_std, 4)}")
    print(f"  Hot  Temp (T=2.0) Probs: {np.round(p_hot, 4)} (Flattened towards uniform 0.20)")
    assert p_cold[2] > p_std[2] > p_hot[2], "Temperature scaling behavior violated!"

    # 3. Top-K Truncation (K = 2)
    _, p_top2 = sample_next_token(step_3_logits, method="top_k", top_k=2)
    print(f"  Top-K (K=2) Distribution: {np.round(p_top2, 6)}")
    assert p_top2[0] == 0.0 and p_top2[1] == 0.0 and p_top2[4] == 0.0, "Top-K non-top tokens must have 0 prob!"
    assert np.isclose(p_top2[2], 0.502139, atol=1e-4) and np.isclose(p_top2[3], 0.497861, atol=1e-4), "Top-K probability mismatch!"

    # 4. Top-P Nucleus (p = 0.70)
    _, p_nucleus = sample_next_token(step_3_logits, method="top_p", top_p=0.70)
    print(f"  Top-P (p=0.70) Nucleus:   {np.round(p_nucleus, 6)}")
    assert p_nucleus[4] == 0.0, "<EOS> should be dropped from nucleus set!"
    assert np.isclose(p_nucleus[2], 0.270426, atol=1e-4), "Top-P probability mismatch!"
    print("  [SUCCESS] All decoding algorithms (Greedy, Temp, Top-K, Top-P) verified against Step-01 derivations!")

    # -------------------------------------------------------------------------
    # CHECKPOINT 4: Autonomous Free-Running Generation & Training Convergence
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[CHECKPOINT 4] TRAINING CONVERGENCE & FREE-RUNNING TEXT GENERATION")
    print("=" * 80)

    # Train model on the sequence to learn the prompt sequence
    print("Training sequence model for 150 epochs on '<BOS> crypto blocks verify <EOS>'...")
    # Full sequence: 0 -> 1 -> 2 -> 3 -> 4
    train_X = np.array([[0, 1, 2, 3]], dtype=np.int64)  # (1, 4)
    train_Y = np.array([[1, 2, 3, 4]], dtype=np.int64)  # (1, 4)

    np.random.seed(42)
    train_params = {
        "E": np.random.randn(5, 3) * 0.1,
        "W_xh": np.random.randn(3, 8) * 0.1,
        "W_hh": np.random.randn(8, 8) * 0.1,
        "b_h": np.zeros((1, 8)),
        "W_hy": np.random.randn(8, 5) * 0.1,
        "b_y": np.zeros((1, 5)),
    }
    train_h0 = np.zeros((1, 8))

    trained_params, loss_hist, ppl_hist = train_sequence_model(
        train_X,
        train_Y,
        train_params,
        train_h0,
        epochs=150,
        learning_rate=0.3,
        verbose=True,
    )

    print(f"\nTraining Results: Initial Loss = {loss_hist[0]:.4f} (PPL = {ppl_hist[0]:.2f}) -> Final Loss = {loss_hist[-1]:.4f} (PPL = {ppl_hist[-1]:.2f})")
    assert loss_hist[-1] < 0.1, "Model failed to overfit target sequence!"
    assert ppl_hist[-1] < 1.1, "Perplexity failed to collapse to near-certainty!"
    print("  [SUCCESS] Loss successfully converged and Perplexity collapsed near 1.0 (near certainty)!")

    # Test autonomous generation
    print("\nAutoregressive Free-Running Generation from prompt '<BOS>':")
    prompt = [0]  # <BOS>
    generated = generate_sequence(
        prompt_tokens=prompt,
        max_new_tokens=4,
        parameters=trained_params,
        method="greedy",
        eos_token_id=4,
    )
    generated_words = [vocab[idx] for idx in generated]
    print(f"  Prompt:    '{vocab[0]}'")
    print(f"  Generated: {' '.join(generated_words)}")
    assert generated == [0, 1, 2, 3, 4], f"Generated tokens {generated} did not match expected [0, 1, 2, 3, 4]!"
    print("  [SUCCESS] Autoregressive model successfully generated '<BOS> crypto blocks verify <EOS>' autonomously!")

    print("\n" + "=" * 80)
    print("ALL VERIFICATION CHECKPOINTS COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")
