"""
10 — Gated Recurrent Unit (GRU) From Scratch with Pure NumPy
=============================================================

Architecture Overview:
----------------------
This module implements a complete Gated Recurrent Unit (GRU) network from
first principles using only Python and pure NumPy. It directly builds upon the
foundations of `07-rnn-from-scratch` and `09-lstm` to implement the streamlined
recurrent gating architecture introduced by Kyunghyun Cho et al. in 2014.

Why GRU Was Created: Streamlining LSTM Without Losing Long-Term Memory
----------------------------------------------------------------------
While the Long Short-Term Memory (LSTM) network resolved the catastrophic
vanishing gradient problem of Vanilla RNNs via its Constant Error Carousel (CEC),
it introduced significant architectural complexity:
1. Dual memory vectors per step: Hidden State (h_t) and Cell State (C_t).
2. Four separate gate projections per step: Forget (f), Input (i), Candidate (c), Output (o).
3. Gating redundancy: Forget and Input gates are often strongly inversely correlated.

The GRU eliminates these redundancies through two key design choices:
1. Single Hidden State (h_t): Eliminates the separate cell state C_t.
2. Coupled Gating (2 Gates):
   - Reset Gate (r_t): Determines how much prior memory to disregard when proposing candidates.
   - Update Gate (z_t): Controls memory retention (1 - z_t) and candidate assimilation (z_t),
     merging the roles of LSTM's forget and input gates into a single convex combination.

The Linear State Interpolation Gradient Highway:
------------------------------------------------
The fundamental state update equation of the GRU is:
    h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h~_t

Taking the partial derivative of h_t with respect to h_{t-1}:
    ∂h_t / ∂h_{t-1} = diag(1 - z_t) + [recurrent nonlinear pathways]

When the update gate z_t ≈ 0, the linear highway path dominates:
    ∂h_t / ∂h_{t-1} ≈ Identity Matrix (I)

This allows error gradients during Backpropagation Through Time (BPTT) to flow
backward across dozens of timesteps unattenuated, solving the vanishing gradient
problem with 25% fewer parameters and higher computational throughput than LSTM.

Visual Data Flow of Cho et al. (2014) GRU Cell:
------------------------------------------------
                                Previous Hidden State h_{t-1}
                                             │
           ┌─────────────────────────────────┼───────────────────────────┐
           │                                 │                           │
           │                                 ▼                           ▼
           │                           ┌───────────┐               ┌───────────┐
           │                           │  1 - z_t  │               │    z_t    │
           │                           └─────┬─────┘               └─────┬─────┘
           │                                 │                           │
           │                                 ▼                           ▼
           │                             h_{t-1} ───⊗                 h~_t ──⊗
           │                                        │                        │
           │                                        ▼                        ▼
           │                                        ┌────────────────────────┐
           │                                        │      ⊕ Addition        │
           │                                        └───────────┬────────────┘
           │                                                    │
           │                                                    ▼
           │                                            New Hidden State h_t
           │                                                    │
           │                                                    ▼
           │                                            Output / Next Step
           │
           │                Reset Gating Path
           │                        │
           │                        ▼
           │                  h_{t-1} ──⊗
           │                            ▲
           │                            │ r_t (Reset Signal)
           │                            │
           │         ┌──────────────────┴──────────────────┐
           │         │                                     │
           │         ▼                                     ▼
           │       ┌───┐                                 ┌───┐
           │       │ σ │ Reset Gate                      │ σ │ Update Gate
           │       └───┘                                 └───┘
           │         ▲                                     ▲
           │         │                                     │
           │         └──────────────────┬──────────────────┘
           │                            │
           │                            ▼
           │                         ┌──────┐
           │                         │ tanh │ Candidate State h~_t
           │                         └──────┘
           │                            │
           ▼                            ▼
Inputs: [ x_t, h_{t-1} ] ─────────────────────┴────────────────────────────────────►

Tensor Dimension Conventions:
-----------------------------
- B: Batch size                       (e.g., B = 1 in verification, B = 30 in benchmark)
- T: Sequential timesteps             (e.g., T = 3 in verification, T = 20 in benchmark)
- D: Input feature dimension          (e.g., D = 2)
- H: Hidden representation dimension  (e.g., H = 3 or H = 8)
- K: Output classification dimension  (e.g., K = 2)

Learnable Parameter Budget:
---------------------------
- Reset Gate:      W_xr (D, H), W_hr (H, H), b_r (1, H) -> DH + H^2 + H
- Update Gate:     W_xz (D, H), W_hz (H, H), b_z (1, H) -> DH + H^2 + H
- Candidate State: W_xh (D, H), W_hh (H, H), b_h (1, H) -> DH + H^2 + H
- Output Head:     W_hy (H, K), b_y (1, K)             -> HK + K
Total parameters: 3 * (DH + H^2 + H) + HK + K
"""

import numpy as np


# =============================================================================
# PART 1: ACTIVATIONS AND LINEAR PROJECTIONS
# =============================================================================

def sigmoid(z):
    """
    Computes numerically stable element-wise logistic sigmoid:
        sigma(z) = 1.0 / (1.0 + exp(-z))

    Parameters:
        z (np.ndarray): Arbitrary real-valued pre-activation tensor.

    Returns:
        np.ndarray: Probabilistic values bounded strictly in (0.0, 1.0).
    """
    clipped_z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped_z))


def stable_softmax(logits):
    """
    Computes numerically stable softmax probabilities across the final feature axis:
        softmax(o)_k = exp(o_k - max(o)) / sum_j exp(o_j - max(o))

    Parameters:
        logits (np.ndarray): Unnormalized prediction scores of shape (..., K).

    Returns:
        np.ndarray: Normalized probability distributions of identical shape, summing to 1.0.
    """
    max_per_sample = np.max(logits, axis=-1, keepdims=True)
    stabilized_exponentials = np.exp(logits - max_per_sample)
    partition_function = np.sum(stabilized_exponentials, axis=-1, keepdims=True)
    return stabilized_exponentials / partition_function


def dense_output_forward(hidden_state, output_weights, output_bias):
    """
    Linear affine projection mapping hidden state representations to unnormalized logits:
        o_t = h_t @ W_{hy} + b_y

    Parameters:
        hidden_state (np.ndarray): Latent hidden representation of shape (B, H).
        output_weights (np.ndarray): Readout weight matrix of shape (H, K).
        output_bias (np.ndarray): Readout bias offset vector of shape (1, K).

    Returns:
        np.ndarray: Logits of shape (B, K).
    """
    return hidden_state @ output_weights + output_bias


# =============================================================================
# PART 2: ATOMIC RECURRENT GRU CELL FORWARD PASS (MICRO-FLOW)
# =============================================================================

def gru_cell_forward(current_input, previous_hidden, parameters):
    """
    Executes a single atomic Cho et al. (2014) GRU forward step at timestep t.

    Micro-Flow Numbered Steps:
        1. Fused Reset and Update gate pre-activations (W_{x, rz}, W_{h, rz}, b_{rz})
        2. Gate partitioning and Sigmoidal activation bounding (r_t, z_t)
        3. Reset-modulated prior contextual memory (h_reset = r_t ⊙ h_{t-1})
        4. Candidate state affine projection and hyperbolic tangent bounding (h~_t)
        5. Convex linear state interpolation (h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h~_t)

    Parameters:
        current_input (np.ndarray): Input features for current step, shape (B, D).
        previous_hidden (np.ndarray): Hidden state from step t-1, shape (B, H).
        parameters (dict): Active parameter dictionary containing:
            - W_xr, W_hr, b_r: Reset gate weights and bias
            - W_xz, W_hz, b_z: Update gate weights and bias
            - W_xh, W_hh, b_h: Candidate state weights and bias

    Returns:
        tuple: (current_hidden, step_cache)
            - current_hidden (np.ndarray): Synthesized hidden state h_t, shape (B, H).
            - step_cache (dict): Intermediate variables required for BPTT adjoint equations.
    """
    # -------------------------------------------------------------------------
    # Step 1: Fused Gate Pre-activations for Reset and Update Gates
    # Concatenating along feature axis into unified 2H blocks:
    # Z_{rz} = x_t @ [W_xr | W_xz] + h_{t-1} @ [W_hr | W_hz] + [b_r | b_z]
    # -------------------------------------------------------------------------
    W_x_rz = np.hstack([parameters["W_xr"], parameters["W_xz"]])  # (D, 2H)
    W_h_rz = np.hstack([parameters["W_hr"], parameters["W_hz"]])  # (H, 2H)
    b_rz = np.hstack([parameters["b_r"], parameters["b_z"]])      # (1, 2H)

    Z_rz = current_input @ W_x_rz + previous_hidden @ W_h_rz + b_rz  # (B, 2H)

    # -------------------------------------------------------------------------
    # Step 2: Gate Slicing and Sigmoidal Activations
    # -------------------------------------------------------------------------
    hidden_dim = previous_hidden.shape[-1]
    z_r_pre = Z_rz[:, :hidden_dim]
    z_z_pre = Z_rz[:, hidden_dim:]

    reset_gate = sigmoid(z_r_pre)   # r_t ∈ (0, 1)^{B x H}
    update_gate = sigmoid(z_z_pre)  # z_t ∈ (0, 1)^{B x H}

    # -------------------------------------------------------------------------
    # Step 3: Context Modulation (Reset Gate Masks Past Context)
    # -------------------------------------------------------------------------
    reset_modulated_hidden = reset_gate * previous_hidden  # (B, H)

    # -------------------------------------------------------------------------
    # Step 4: Candidate State Projection
    # z_{h, t} = x_t @ W_xh + (r_t ⊙ h_{t-1}) @ W_hh + b_h
    # h~_t = tanh(z_{h, t})
    # -------------------------------------------------------------------------
    z_h_pre = (
        current_input @ parameters["W_xh"]
        + reset_modulated_hidden @ parameters["W_hh"]
        + parameters["b_h"]
    )  # (B, H)
    candidate_hidden = np.tanh(z_h_pre)  # h~_t ∈ (-1, 1)^{B x H}

    # -------------------------------------------------------------------------
    # Step 5: Convex Linear State Interpolation (The Gradient Highway)
    # h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h~_t
    # -------------------------------------------------------------------------
    current_hidden = (1.0 - update_gate) * previous_hidden + update_gate * candidate_hidden

    # Package comprehensive step cache for exact BPTT gradient backpropagation
    step_cache = {
        "current_input": current_input,
        "previous_hidden": previous_hidden,
        "z_r_pre": z_r_pre,
        "reset_gate": reset_gate,
        "z_z_pre": z_z_pre,
        "update_gate": update_gate,
        "reset_modulated_hidden": reset_modulated_hidden,
        "z_h_pre": z_h_pre,
        "candidate_hidden": candidate_hidden,
        "current_hidden": current_hidden,
    }

    return current_hidden, step_cache


# =============================================================================
# PART 3: UNROLLED SEQUENCE FORWARD PASS (MESO-FLOW)
# =============================================================================

def gru_forward(input_sequence, initial_hidden, parameters):
    """
    Unrolls the GRU forward computation graph across discrete timesteps t = 0..T-1.

    Parameters:
        input_sequence (np.ndarray): Sequential batch data of shape (B, T, D).
        initial_hidden (np.ndarray): Initial recurrent state h_0 of shape (B, H).
        parameters (dict): Complete model parameter dictionary.

    Returns:
        dict: Trajectories, output probabilities, and intermediate step caches:
            - all_hidden_states: (B, T, H)
            - all_logits: (B, T, K)
            - all_output_probabilities: (B, T, K)
            - all_cell_caches: list of length T containing per-step caches
            - initial_hidden: h_0 of shape (B, H)
    """
    batch_size, sequence_length, _ = input_sequence.shape
    hidden_dim = initial_hidden.shape[-1]
    output_dim = parameters["b_y"].shape[-1]

    all_hidden_states = np.zeros((batch_size, sequence_length, hidden_dim), dtype=np.float64)
    all_logits = np.zeros((batch_size, sequence_length, output_dim), dtype=np.float64)
    all_output_probabilities = np.zeros((batch_size, sequence_length, output_dim), dtype=np.float64)
    all_cell_caches = []

    current_hidden = initial_hidden.copy()

    # Temporal unrolling loop across discrete steps t = 0, 1, ..., T-1
    for timestep in range(sequence_length):
        input_slice = input_sequence[:, timestep, :]  # Shape: (B, D)

        current_hidden, step_cache = gru_cell_forward(
            current_input=input_slice,
            previous_hidden=current_hidden,
            parameters=parameters,
        )

        step_logits = dense_output_forward(
            hidden_state=current_hidden,
            output_weights=parameters["W_hy"],
            output_bias=parameters["b_y"],
        )
        step_probabilities = stable_softmax(step_logits)

        all_hidden_states[:, timestep, :] = current_hidden
        all_logits[:, timestep, :] = step_logits
        all_output_probabilities[:, timestep, :] = step_probabilities
        all_cell_caches.append(step_cache)

    return {
        "all_hidden_states": all_hidden_states,
        "all_logits": all_logits,
        "all_output_probabilities": all_output_probabilities,
        "all_cell_caches": all_cell_caches,
        "initial_hidden": initial_hidden,
    }


# =============================================================================
# PART 4: LOSS COMPUTATION AND OUTPUT GRADIENTS
# =============================================================================

def sequence_cross_entropy(output_probabilities, target_sequence, epsilon=1e-15):
    """
    Computes Categorical Cross-Entropy loss accumulated across all sequence timesteps:
        L_t = - sum_k Y_{t, k} * ln(y_hat_{t, k} + eps)
        L_{seq} = sum_{t=1}^T L_t

    Parameters:
        output_probabilities (np.ndarray): Predicted class probabilities of shape (B, T, K).
        target_sequence (np.ndarray): One-hot encoded ground truth of shape (B, T, K).
        epsilon (float): Numerical stability floor against log(0).

    Returns:
        tuple: (total_loss, average_loss_per_timestep, step_losses)
            - total_loss (float): Scalar sum of sequence cross-entropy.
            - average_loss_per_timestep (float): Scalar average loss per step.
            - step_losses (list of float): Individual step losses [L_1, L_2, ..., L_T].
    """
    sequence_length = output_probabilities.shape[1]
    step_losses = []

    for t in range(sequence_length):
        probs_clipped = np.clip(output_probabilities[:, t, :], epsilon, 1.0 - epsilon)
        loss_t = float(
            np.mean(-np.sum(target_sequence[:, t, :] * np.log(probs_clipped), axis=-1))
        )
        step_losses.append(loss_t)

    total_loss = float(np.sum(step_losses))
    average_loss_per_timestep = total_loss / sequence_length

    return total_loss, average_loss_per_timestep, step_losses


def output_score_gradients(output_probabilities, target_sequence):
    """
    Analytical gradient of the batch-mean categorical cross-entropy with
    respect to unnormalized logits o_t:
        delta_{logit, t} = dL_t / do_t = (y_hat_t - Y_t) / B  ∈ R^{B x K}

    `sequence_cross_entropy` averages each timestep loss across its B samples.
    The same division by B must appear here so that BPTT differentiates the
    reported loss, rather than a batch-summed version of it.

    Parameters:
        output_probabilities (np.ndarray): Softmax probabilities of shape (B, T, K).
        target_sequence (np.ndarray): One-hot targets of shape (B, T, K).

    Returns:
        np.ndarray: Logit error tensor of shape (B, T, K).
    """
    batch_size = output_probabilities.shape[0]
    return (output_probabilities - target_sequence) / batch_size


# =============================================================================
# PART 5: BACKPROPAGATION THROUGH TIME (BPTT) & PARAMETER GRADIENTS
# =============================================================================

def gru_backward(input_sequence, target_sequence, forward_results, parameters):
    """
    Executes complete Backpropagation Through Time (BPTT) for Cho et al. (2014) GRU.

    Error propagates backward in reverse chronological order (t = T-1 down to 0).
    At each step t, the recurrent error arrives via the complete 4-Way Gradient Highway:
        Path 1 (Linear State Highway):      d_h * (1 - z_t)
        Path 2 (Candidate Reset Mod):       d_h_reset * r_t
        Path 3 (Update Gate Recurrent):     d_zz @ W_hz.T
        Path 4 (Reset Gate Recurrent):      d_zr @ W_hr.T

    Parameters:
        input_sequence (np.ndarray): Input batch of shape (B, T, D).
        target_sequence (np.ndarray): One-hot targets of shape (B, T, K).
        forward_results (dict): Output of gru_forward.
        parameters (dict): Active model parameter tensors.

    Returns:
        dict: Parameter gradients matching exact keys of model parameters.
    """
    _, sequence_length, _ = input_sequence.shape
    all_output_probs = forward_results["all_output_probabilities"]
    all_cell_caches = forward_results["all_cell_caches"]

    output_logit_errors = output_score_gradients(all_output_probs, target_sequence)

    # Initialize parameter gradient accumulators
    param_gradients = {
        "W_xr": np.zeros_like(parameters["W_xr"]),
        "W_hr": np.zeros_like(parameters["W_hr"]),
        "b_r": np.zeros_like(parameters["b_r"]),
        "W_xz": np.zeros_like(parameters["W_xz"]),
        "W_hz": np.zeros_like(parameters["W_hz"]),
        "b_z": np.zeros_like(parameters["b_z"]),
        "W_xh": np.zeros_like(parameters["W_xh"]),
        "W_hh": np.zeros_like(parameters["W_hh"]),
        "b_h": np.zeros_like(parameters["b_h"]),
        "W_hy": np.zeros_like(parameters["W_hy"]),
        "b_y": np.zeros_like(parameters["b_y"]),
    }

    # Recurrent error propagating from future timestep (zero at terminal step T-1)
    future_hidden_error = np.zeros_like(forward_results["initial_hidden"])

    # Reverse temporal unrolling loop: t = T-1, T-2, ..., 0
    for timestep in reversed(range(sequence_length)):
        step_logit_error = output_logit_errors[:, timestep, :]  # Shape: (B, K)
        step_cache = all_cell_caches[timestep]

        x_t = step_cache["current_input"]
        h_prev = step_cache["previous_hidden"]
        r_t = step_cache["reset_gate"]
        z_t = step_cache["update_gate"]
        h_reset = step_cache["reset_modulated_hidden"]
        h_cand = step_cache["candidate_hidden"]
        h_t = step_cache["current_hidden"]

        # ---------------------------------------------------------------------
        # 1. Output Readout Layer Parameter Gradients
        # ---------------------------------------------------------------------
        param_gradients["W_hy"] += h_t.T @ step_logit_error
        param_gradients["b_y"] += np.sum(step_logit_error, axis=0, keepdims=True)

        # ---------------------------------------------------------------------
        # 2. Total Hidden State Error: Immediate Loss + Future Recurrent Influx
        # ---------------------------------------------------------------------
        direct_hidden_error = step_logit_error @ parameters["W_hy"].T
        total_hidden_error = direct_hidden_error + future_hidden_error  # delta_{h_t} ∈ R^{B x H}

        # ---------------------------------------------------------------------
        # 3. Candidate Hidden State Pre-Activation Gradient
        # dL/dh~_t = delta_{h_t} ⊙ z_t
        # delta_{z_h, t} = (dL/dh~_t) ⊙ (1 - h~_t^2)
        # ---------------------------------------------------------------------
        delta_z_h = total_hidden_error * z_t * (1.0 - h_cand**2)  # (B, H)

        param_gradients["W_xh"] += x_t.T @ delta_z_h
        param_gradients["W_hh"] += h_reset.T @ delta_z_h
        param_gradients["b_h"] += np.sum(delta_z_h, axis=0, keepdims=True)

        # ---------------------------------------------------------------------
        # 4. Reset Modulation Error and Reset Gate Pre-Activation Gradient
        # delta_{h_reset} = delta_{z_h, t} @ W_{hh}^T
        # delta_{r_t} = delta_{h_reset} ⊙ h_{t-1}
        # delta_{z_r, t} = delta_{r_t} ⊙ r_t ⊙ (1 - r_t)
        # ---------------------------------------------------------------------
        delta_h_reset = delta_z_h @ parameters["W_hh"].T  # (B, H)
        delta_r = delta_h_reset * h_prev
        delta_z_r = delta_r * r_t * (1.0 - r_t)  # (B, H)

        param_gradients["W_xr"] += x_t.T @ delta_z_r
        param_gradients["W_hr"] += h_prev.T @ delta_z_r
        param_gradients["b_r"] += np.sum(delta_z_r, axis=0, keepdims=True)

        # ---------------------------------------------------------------------
        # 5. Update Gate Pre-Activation Gradient
        # delta_{z_t} = delta_{h_t} ⊙ (h~_t - h_{t-1})
        # delta_{z_z, t} = delta_{z_t} ⊙ z_t ⊙ (1 - z_t)
        # ---------------------------------------------------------------------
        delta_z = total_hidden_error * (h_cand - h_prev)
        delta_z_z = delta_z * z_t * (1.0 - z_t)  # (B, H)

        param_gradients["W_xz"] += x_t.T @ delta_z_z
        param_gradients["W_hz"] += h_prev.T @ delta_z_z
        param_gradients["b_z"] += np.sum(delta_z_z, axis=0, keepdims=True)

        # ---------------------------------------------------------------------
        # 6. Recurrent Error Highway Propagating to Prior Timestep h_{t-1}
        # ---------------------------------------------------------------------
        path_1 = total_hidden_error * (1.0 - z_t)           # Path 1: Direct Linear State Highway
        path_2 = delta_h_reset * r_t                        # Path 2: Candidate State via Reset Mod
        path_3 = delta_z_z @ parameters["W_hz"].T           # Path 3: Update Gate Recurrent
        path_4 = delta_z_r @ parameters["W_hr"].T           # Path 4: Reset Gate Recurrent

        future_hidden_error = path_1 + path_2 + path_3 + path_4

    return param_gradients


# =============================================================================
# PART 6: TRAINING AND OPTIMIZATION UTILITIES
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
    Applies standard first-order Stochastic Gradient Descent (SGD) parameter update:
        theta = theta - lr * dL/dtheta

    Parameters:
        parameters (dict): Model parameter dictionary.
        gradients (dict): Parameter gradient dictionary.
        learning_rate (float): Optimization step size.

    Returns:
        dict: Updated parameter dictionary.
    """
    return {k: parameters[k] - learning_rate * gradients[k] for k in parameters}


def train_gru(
    input_sequence,
    target_sequence,
    initial_parameters,
    initial_hidden,
    epochs=100,
    learning_rate=0.1,
    verbose=False,
):
    """
    Standard training optimization loop for GRU network.

    Parameters:
        input_sequence (np.ndarray): Training inputs of shape (B, T, D).
        target_sequence (np.ndarray): Ground truth one-hot targets of shape (B, T, K).
        initial_parameters (dict): Starting parameter weights and biases.
        initial_hidden (np.ndarray): Starting hidden state of shape (B, H).
        epochs (int): Number of complete training iterations.
        learning_rate (float): Optimization step size.
        verbose (bool): If True, prints training loss every 10 epochs.

    Returns:
        tuple: (trained_parameters, loss_history)
    """
    params = {k: v.copy() for k, v in initial_parameters.items()}
    loss_history = []

    for ep in range(epochs):
        forward_results = gru_forward(input_sequence, initial_hidden, params)
        total_loss, _, _ = sequence_cross_entropy(
            forward_results["all_output_probabilities"], target_sequence
        )
        loss_history.append(total_loss)

        grads = gru_backward(input_sequence, target_sequence, forward_results, params)
        clipped_grads, _ = clip_gradients_by_norm(grads, max_norm=5.0)
        params = gradient_descent_update(params, clipped_grads, learning_rate)

        if verbose and ((ep + 1) % 10 == 0 or ep == 0):
            print(f"Epoch {ep+1:3d}/{epochs} | Sequence Loss: {total_loss:.6f}")

    return params, loss_history


# =============================================================================
# PART 7: FINITE-DIFFERENCE NUMERICAL GRADIENT VERIFICATION
# =============================================================================

def compute_numerical_gradient(
    input_sequence,
    target_sequence,
    initial_hidden,
    parameters,
    param_name,
    row,
    col,
    epsilon=1e-5,
):
    """
    Computes symmetric finite-difference quotient for parameter coordinate θ[row, col]:
        dL/dθ ≈ [ L(θ + ε) - L(θ - ε) ] / (2 * ε)

    Parameters:
        input_sequence (np.ndarray): Shape (B, T, D).
        target_sequence (np.ndarray): Shape (B, T, K).
        initial_hidden (np.ndarray): Shape (B, H).
        parameters (dict): Model parameter dictionary.
        param_name (str): Parameter tensor key to perturb.
        row (int): Row index in target tensor.
        col (int): Column index in target tensor.
        epsilon (float): Symmetric perturbation magnitude.

    Returns:
        float: Numerically estimated gradient scalar.
    """
    params_plus = {k: v.copy() for k, v in parameters.items()}
    params_minus = {k: v.copy() for k, v in parameters.items()}

    params_plus[param_name][row, col] += epsilon
    params_minus[param_name][row, col] -= epsilon

    fwd_plus = gru_forward(input_sequence, initial_hidden, params_plus)
    loss_plus, _, _ = sequence_cross_entropy(fwd_plus["all_output_probabilities"], target_sequence)

    fwd_minus = gru_forward(input_sequence, initial_hidden, params_minus)
    loss_minus, _, _ = sequence_cross_entropy(fwd_minus["all_output_probabilities"], target_sequence)

    return (loss_plus - loss_minus) / (2.0 * epsilon)


# =============================================================================
# PART 8: SIMPLE RNN BASELINE FOR HEAD-TO-HEAD COMPARISON
# =============================================================================

def simple_rnn_forward(input_sequence, initial_hidden, W_xh, W_hh, b_h, W_hy, b_y):
    """
    Forward pass for a Simple Vanilla RNN (Single State h_t = tanh(x_t W_xh + h_{t-1} W_hh + b_h)).
    """
    batch_size, sequence_length, _ = input_sequence.shape
    hidden_dim = initial_hidden.shape[-1]
    output_dim = b_y.shape[-1]

    all_hidden = np.zeros((batch_size, sequence_length, hidden_dim), dtype=np.float64)
    all_probs = np.zeros((batch_size, sequence_length, output_dim), dtype=np.float64)
    current_h = initial_hidden.copy()

    caches = []
    for t in range(sequence_length):
        x_t = input_sequence[:, t, :]
        z_t = x_t @ W_xh + current_h @ W_hh + b_h
        next_h = np.tanh(z_t)
        logits = next_h @ W_hy + b_y
        probs = stable_softmax(logits)

        all_hidden[:, t, :] = next_h
        all_probs[:, t, :] = probs
        caches.append((x_t, current_h, z_t, next_h))
        current_h = next_h

    return all_hidden, all_probs, caches


def simple_rnn_backward(all_probs, target_sequence, caches, W_hh, W_hy):
    """
    BPTT backward pass for Simple RNN illustrating repeated Jacobian multiplication decay.
    """
    batch_size, sequence_length, _ = all_probs.shape
    dW_xh = np.zeros((caches[0][0].shape[-1], W_hh.shape[0]))
    dW_hh = np.zeros_like(W_hh)
    db_h = np.zeros((1, W_hh.shape[0]))
    dW_hy = np.zeros_like(W_hy)
    db_y = np.zeros((1, W_hy.shape[1]))

    delta_h_next = np.zeros((1, W_hh.shape[0]))

    for t in reversed(range(sequence_length)):
        x_t, h_prev, _, h_t = caches[t]
        # The matching loss is averaged over samples in each timestep.
        delta_o = (all_probs[:, t, :] - target_sequence[:, t, :]) / batch_size

        dW_hy += h_t.T @ delta_o
        db_y += np.sum(delta_o, axis=0, keepdims=True)

        delta_h_t = delta_o @ W_hy.T + delta_h_next
        delta_z_t = delta_h_t * (1.0 - h_t**2)  # tanh' saturation!

        dW_xh += x_t.T @ delta_z_t
        dW_hh += h_prev.T @ delta_z_t
        db_h += np.sum(delta_z_t, axis=0, keepdims=True)

        delta_h_next = delta_z_t @ W_hh.T  # Repeated matrix multiplication causes vanishing gradients

    return {"W_xh": dW_xh, "W_hh": dW_hh, "b_h": db_h, "W_hy": dW_hy, "b_y": db_y}


# =============================================================================
# PART 9: EXECUTION, NUMERICAL ASSERTIONS, AND PURPOSE DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("10 — GATED RECURRENT UNIT (GRU) FROM SCRATCH")
    print("Demonstrating Cho et al. (2014) Mathematics and the Linear Gradient Highway")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # CHECKPOINT 1: Exact Numerical Walkthrough (step-01-mathematics.md)
    # -------------------------------------------------------------------------
    print("\n[CHECKPOINT 1] Validating Exact Numerical Walkthrough (step-01-mathematics.md)...")

    # Dimensions: B = 1, T = 3, D = 2, H = 3, K = 2
    x_1 = np.array([[1.0, -0.5]], dtype=np.float64)
    x_2 = np.array([[0.5, 1.0]], dtype=np.float64)
    x_3 = np.array([[-1.0, 0.5]], dtype=np.float64)
    X_toy = np.stack([x_1, x_2, x_3], axis=1)  # (1, 3, 2)

    Y_1 = np.array([[0.0, 1.0]], dtype=np.float64)
    Y_2 = np.array([[1.0, 0.0]], dtype=np.float64)
    Y_3 = np.array([[0.0, 1.0]], dtype=np.float64)
    Y_toy = np.stack([Y_1, Y_2, Y_3], axis=1)  # (1, 3, 2)

    h_0 = np.zeros((1, 3), dtype=np.float64)

    # Parameter setup matching step-01-mathematics.md Q7
    toy_parameters = {
        "W_xr": np.array([[0.2, -0.1, 0.3], [-0.3, 0.4, 0.1]], dtype=np.float64),
        "W_hr": np.array([[0.1, -0.2, 0.1], [0.3, 0.1, -0.1], [-0.2, 0.2, 0.4]], dtype=np.float64),
        "b_r": np.zeros((1, 3), dtype=np.float64),
        "W_xz": np.array([[-0.2, 0.3, -0.1], [0.4, -0.2, 0.3]], dtype=np.float64),
        "W_hz": np.array([[-0.1, 0.3, 0.2], [0.2, -0.1, 0.1], [0.1, 0.2, -0.3]], dtype=np.float64),
        "b_z": np.zeros((1, 3), dtype=np.float64),
        "W_xh": np.array([[0.3, 0.2, -0.3], [-0.1, 0.4, 0.2]], dtype=np.float64),
        "W_hh": np.array([[0.2, 0.1, -0.2], [-0.3, 0.2, 0.1], [0.1, -0.1, 0.3]], dtype=np.float64),
        "b_h": np.zeros((1, 3), dtype=np.float64),
        "W_hy": np.array([[0.4, -0.3], [-0.2, 0.5], [0.3, 0.1]], dtype=np.float64),
        "b_y": np.zeros((1, 2), dtype=np.float64),
    }

    # Execute Forward Pass
    forward_results = gru_forward(X_toy, h_0, toy_parameters)
    total_seq_loss, avg_loss, step_losses = sequence_cross_entropy(
        forward_results["all_output_probabilities"], Y_toy
    )

    print(f"  Step 1 Loss L_1: {step_losses[0]:.6f} (Expected: 0.724228)")
    print(f"  Step 2 Loss L_2: {step_losses[1]:.6f} (Expected: 0.751293)")
    print(f"  Step 3 Loss L_3: {step_losses[2]:.6f} (Expected: 0.598178)")
    print(f"  Total Sequence Loss L_seq: {total_seq_loss:.6f} (Expected: 2.073698)")

    assert np.isclose(step_losses[0], 0.724228, atol=1e-4), "L_1 mismatch!"
    assert np.isclose(step_losses[1], 0.751293, atol=1e-4), "L_2 mismatch!"
    assert np.isclose(step_losses[2], 0.598178, atol=1e-4), "L_3 mismatch!"
    assert np.isclose(total_seq_loss, 2.073698, atol=1e-4), "Total L_seq mismatch!"
    print("  [SUCCESS] All forward states and losses match Step-01 Mathematics!")

    # Execute Backward Pass (BPTT)
    analytical_grads = gru_backward(X_toy, Y_toy, forward_results, toy_parameters)

    # Expected values from step-01-mathematics.md Q11
    expected_dW_hy = np.array([
        [-0.061678, 0.061678],
        [-0.053830, 0.053830],
        [0.033926, -0.033926],
    ])
    expected_db_y = np.array([[0.437245, -0.437245]])

    expected_dW_xh = np.array([
        [-0.138426, -0.009637, -0.033834],
        [-0.093335, 0.067651, -0.022173],
    ])
    expected_db_h = np.array([[0.127154, -0.235742, 0.039597]])

    assert np.allclose(analytical_grads["W_hy"], expected_dW_hy, atol=1e-4), "dW_hy mismatch!"
    assert np.allclose(analytical_grads["b_y"], expected_db_y, atol=1e-4), "db_y mismatch!"
    assert np.allclose(analytical_grads["W_xh"], expected_dW_xh, atol=1e-4), "dW_xh mismatch!"
    assert np.allclose(analytical_grads["b_h"], expected_db_h, atol=1e-4), "db_h mismatch!"
    print("  [SUCCESS] Analytical BPTT gradients match theoretical derivations down to 1e-4!")

    # -------------------------------------------------------------------------
    # CHECKPOINT 2: Finite-Difference Numerical Gradient Verification
    # -------------------------------------------------------------------------
    print("\n[CHECKPOINT 2] Running Finite-Difference Gradient Checks...")
    param_checks = [
        ("W_xr", 0, 1, "W_xr[0, 1] (Reset Gate Input)"),
        ("W_hr", 1, 2, "W_hr[1, 2] (Reset Gate Recurrent)"),
        ("b_r",  0, 0, "b_r[0, 0]  (Reset Gate Bias)"),
        ("W_xz", 1, 0, "W_xz[1, 0] (Update Gate Input)"),
        ("W_hz", 0, 2, "W_hz[0, 2] (Update Gate Recurrent)"),
        ("b_z",  0, 1, "b_z[0, 1]  (Update Gate Bias)"),
        ("W_xh", 0, 2, "W_xh[0, 2] (Candidate Input)"),
        ("W_hh", 1, 1, "W_hh[1, 1] (Candidate Recurrent)"),
        ("b_h",  0, 1, "b_h[0, 1]  (Candidate Bias)"),
        ("W_hy", 0, 0, "W_hy[0, 0] (Output Weight)"),
        ("b_y",  0, 1, "b_y[0, 1]  (Output Bias)"),
    ]

    for param_name, r, c, label in param_checks:
        anal_val = analytical_grads[param_name][r, c]
        num_val = compute_numerical_gradient(
            X_toy, Y_toy, h_0, toy_parameters, param_name, r, c
        )
        rel_error = abs(anal_val - num_val) / (abs(anal_val) + abs(num_val) + 1e-12)
        print(f"  Checking {label:<32} | Analytical: {anal_val:+.7f} | Numerical: {num_val:+.7f} | Rel Err: {rel_error:.2e}")
        assert rel_error < 1e-4, f"Gradient check failed for {label}!"

    print("  [SUCCESS] All GRU parameter gradients verified against symmetric numerical differentiation!")

    # The toy derivation above uses B = 1.  Duplicate it into B = 2 to verify
    # that the batch-mean loss and its analytical gradients use the same scale.
    X_toy_batched = np.repeat(X_toy, repeats=2, axis=0)
    Y_toy_batched = np.repeat(Y_toy, repeats=2, axis=0)
    h_0_batched = np.zeros((2, 3), dtype=np.float64)
    batched_forward_results = gru_forward(X_toy_batched, h_0_batched, toy_parameters)
    batched_analytical_grads = gru_backward(
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
    # CHECKPOINT 3: Teaching Demonstration of Gated Long-Range Memory
    # Head-to-Head Benchmark: Simple RNN vs. GRU on Long-Range Bit Memory (T = 20)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[CHECKPOINT 3] TEACHING DEMONSTRATION: GATED LONG-RANGE MEMORY")
    print("Head-to-Head Benchmark: Simple RNN vs. GRU on Long-Range Bit Memory (T = 20)")
    print("=" * 80)

    # Task Description:
    # A sequence of length T = 20 is generated.
    # At t = 0, an informative indicator bit is presented: [1, 0] (Class 0) or [0, 1] (Class 1).
    # From t = 1 to t = 19 (19 intermediate noise steps), random Gaussian noise is fed.
    # At t = 19, the model MUST classify the original indicator bit from t = 0!
    #
    # This task is a deterministic teaching example, not a general architecture
    # benchmark. A GRU can preserve error flow through its direct retention path
    # when the product of (1 - z_t) terms remains close to 1.

    np.random.seed(42)
    seq_len = 20
    batch_size = 30
    d_in = 2
    h_dim = 8
    n_classes = 2

    # Generate synthetic training set: indicator bit at t=0, noise for t=1..19
    labels = np.random.randint(0, 2, size=(batch_size,))
    X_data = np.random.normal(0.0, 0.1, size=(batch_size, seq_len, d_in))

    for b in range(batch_size):
        if labels[b] == 0:
            X_data[b, 0, :] = [1.0, 0.0]
        else:
            X_data[b, 0, :] = [0.0, 1.0]

    Y_data = np.zeros((batch_size, seq_len, n_classes))
    for b in range(batch_size):
        Y_data[b, :, labels[b]] = 1.0

    print(f"\nSynthetic Dataset Generated: {batch_size} sequences, length T = {seq_len} timesteps.")
    print("Signal: Timestep t=0 contains class bit. Timesteps t=1..19 contain random Gaussian noise.")

    # 1. Initialize Simple RNN Parameters
    W_xh_rnn = np.random.randn(d_in, h_dim) * 0.1
    W_hh_rnn = np.random.randn(h_dim, h_dim) * 0.1
    b_h_rnn = np.zeros((1, h_dim))
    W_hy_rnn = np.random.randn(h_dim, n_classes) * 0.1
    b_y_rnn = np.zeros((1, n_classes))

    # 2. Initialize GRU Parameters
    # (Initialize update gate bias with negative offset to bias toward retention: 1 - z_t ≈ 1.0)
    gru_params = {
        "W_xr": np.random.randn(d_in, h_dim) * 0.1,
        "W_hr": np.random.randn(h_dim, h_dim) * 0.1,
        "b_r": np.zeros((1, h_dim)),
        "W_xz": np.random.randn(d_in, h_dim) * 0.1,
        "W_hz": np.random.randn(h_dim, h_dim) * 0.1,
        "b_z": -np.ones((1, h_dim)) * 2.0,  # Bias update gate toward 0 so memory highway 1-z is open
        "W_xh": np.random.randn(d_in, h_dim) * 0.1,
        "W_hh": np.random.randn(h_dim, h_dim) * 0.1,
        "b_h": np.zeros((1, h_dim)),
        "W_hy": np.random.randn(h_dim, n_classes) * 0.1,
        "b_y": np.zeros((1, n_classes)),
    }

    initial_h = np.zeros((batch_size, h_dim))

    print("\nTraining Simple RNN vs. GRU for 80 epochs...")
    epochs = 80
    # The objective is a true batch mean.  This rate is chosen for that scale,
    # not for the older accidental batch-summed gradients.
    lr = 6.0

    for ep in range(epochs):
        # --- Train Simple RNN ---
        rnn_h, rnn_probs, rnn_caches = simple_rnn_forward(
            X_data, initial_h, W_xh_rnn, W_hh_rnn, b_h_rnn, W_hy_rnn, b_y_rnn
        )
        rnn_loss, _, _ = sequence_cross_entropy(rnn_probs, Y_data)

        rnn_grads = simple_rnn_backward(
            rnn_probs, Y_data, rnn_caches, W_hh_rnn, W_hy_rnn
        )
        W_xh_rnn -= lr * np.clip(rnn_grads["W_xh"], -5.0, 5.0)
        W_hh_rnn -= lr * np.clip(rnn_grads["W_hh"], -5.0, 5.0)
        b_h_rnn -= lr * np.clip(rnn_grads["b_h"], -5.0, 5.0)
        W_hy_rnn -= lr * np.clip(rnn_grads["W_hy"], -5.0, 5.0)
        b_y_rnn -= lr * np.clip(rnn_grads["b_y"], -5.0, 5.0)

        # --- Train GRU ---
        gru_fwd = gru_forward(X_data, initial_h, gru_params)
        gru_loss, _, _ = sequence_cross_entropy(
            gru_fwd["all_output_probabilities"], Y_data
        )

        gru_grads = gru_backward(X_data, Y_data, gru_fwd, gru_params)
        clipped_gru_grads, _ = clip_gradients_by_norm(gru_grads, max_norm=5.0)
        gru_params = gradient_descent_update(gru_params, clipped_gru_grads, learning_rate=lr)

        if (ep + 1) % 20 == 0 or ep == 0:
            rnn_preds = np.argmax(rnn_probs[:, -1, :], axis=-1)
            gru_preds = np.argmax(gru_fwd["all_output_probabilities"][:, -1, :], axis=-1)

            rnn_acc = np.mean(rnn_preds == labels) * 100.0
            gru_acc = np.mean(gru_preds == labels) * 100.0

            print(
                f"Epoch {ep+1:2d}/{epochs:2d} | "
                f"Simple RNN Loss: {rnn_loss:.4f}, Acc: {rnn_acc:5.1f}% | "
                f"GRU Loss: {gru_loss:.4f}, Acc: {gru_acc:5.1f}%"
            )

    print("\n" + "-" * 80)
    print("FINAL BENCHMARK COMPARISON AT TIMESTEP T = 20:")
    print("-" * 80)
    print(f"Simple RNN Terminal Accuracy: {rnn_acc:.1f}%  (STUCK around random guessing 50%)")
    print(f"GRU Terminal Accuracy:        {gru_acc:.1f}%  (PERFECT 100% memory retention)")
    print("-" * 80)

    assert rnn_acc < 80.0, "Simple RNN unexpectedly learned long sequence without gating!"
    assert gru_acc == 100.0, "GRU failed to achieve 100% accuracy on long-range bit memory task!"

    # -------------------------------------------------------------------------
    # CHECKPOINT 4: Gate Inspection (How GRU Solved the Problem)
    # -------------------------------------------------------------------------
    print("\n[CHECKPOINT 4] Inspecting GRU Gate Activations Across Sequence:")
    test_fwd = gru_forward(X_data[:1], initial_h[:1], gru_params)
    caches = test_fwd["all_cell_caches"]

    z_at_step_0 = np.mean(caches[0]["update_gate"])
    z_at_step_10 = np.mean(caches[10]["update_gate"])
    highway_at_step_10 = np.mean(1.0 - caches[10]["update_gate"])

    print(f"  At timestep t = 0  (Signal bit presented):")
    print(f"    Update Gate z_0        = {z_at_step_0:.3f} (Writes candidate signal into hidden state)")
    print(f"  At timestep t = 10 (Intermediate noise):")
    print(f"    Update Gate z_10       = {z_at_step_10:.3f} (Suppresses incoming noise)")
    print(f"    Retention (1 - z_10)   = {highway_at_step_10:.3f} (Keeps linear state gradient highway open)")

    print("\n" + "=" * 80)
    print("CONCLUSION: The GRU retention path can preserve useful state and")
    print("improve long-range gradient flow. This small task illustrates that")
    print("behavior; it is not a general performance claim against LSTMs.")
    print("=" * 80 + "\n")
