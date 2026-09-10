"""
07 — Recurrent Neural Network (RNN) From Scratch with Pure NumPy
================================================================

Architecture Overview:
----------------------
This module implements a vanilla Many-to-Many Recurrent Neural Network (RNN)
from first principles using only Python and NumPy. It trains on a classic
educational sequence modeling problem: character-level next-step prediction
on the word "hello" (input sequence: "hell" -> target sequence: "ello").

Mathematical Recurrence Equations:
----------------------------------
For each timestep t in 1, ..., T:
    1. Input Projection:
       z_{input, t} = x_t @ W_{xh}
    2. Recurrent Memory Projection:
       z_{recurrent, t} = h_{t-1} @ W_{hh}
    3. Hidden Pre-activation:
       z_t = z_{input, t} + z_{recurrent, t} + b_h
           = x_t @ W_{xh} + h_{t-1} @ W_{hh} + b_h
    4. Non-linear Activation (Bounded Memory):
       h_t = tanh(z_t)
    5. Output Logits (Linear Projection):
       o_t = h_t @ W_{hy} + b_y
    6. Normalized Class Probabilities:
       y_hat_t = softmax(o_t)
    7. Categorical Cross-Entropy Loss:
       L_t = - sum_{k} Y_{t, k} * ln(y_hat_{t, k} + epsilon)
       L_{seq} = sum_{t=1}^T L_t

Tensor Dimension Conventions:
-----------------------------
- B: Batch size                       (B = 1 in educational demo)
- T: Number of sequential timesteps   (T = 4 for "hell")
- D: Input feature dimension          (D = 4 one-hot character vector size)
- H: Hidden recurrent state dimension (H = 3 internal memory neurons)
- K: Output classification dimension  (K = 4 vocabulary classes: 'h','e','l','o')

Learnable Parameter Budget (40 parameters total):
-------------------------------------------------
- W_{xh} (input_to_hidden_weights):     shape (D, H) = (4, 3) -> 12 weights
- W_{hh} (hidden_to_hidden_weights):    shape (H, H) = (3, 3) ->  9 weights
- b_h    (hidden_bias):                 shape (1, H) = (1, 3) ->  3 biases
- W_{hy} (hidden_to_output_weights):    shape (H, K) = (3, 4) -> 12 weights
- b_y    (output_bias):                 shape (1, K) = (1, 4) ->  4 biases
"""

import numpy as np


# =============================================================================
# PART 1: FORWARD PASS (RECURRENT CELL, SOFTMAX, AND UNROLLED SEQUENCE)
# =============================================================================

def rnn_cell_forward(
    current_input,
    previous_hidden_state,
    input_to_hidden_weights,
    hidden_to_hidden_weights,
    hidden_bias,
):
    """
    Computes a single recurrent step: input and prior memory combined into a new hidden state.

    Mathematical Formulation:
        1. Linear Pre-activation:
           z_t = x_t @ W_{xh} + h_{t-1} @ W_{hh} + b_h
        2. Non-linear Bounding (tanh):
           h_t = tanh(z_t)

    Data Flow Diagram:
        current character x_t ────────┐
                                      ├──► raw hidden memory z_t ──► tanh ──► new memory h_t
        previous memory h_{t-1} ──────┘

    Why tanh is used:
        The tanh activation function bounds all recurrent memory states strictly
        between -1.0 and +1.0. This prevents values from exploding exponentially
        as hidden states are passed repeatedly through time.

    Parameters:
        current_input (np.ndarray):
            Input slice x_t at timestep t.
            Shape: (B, D) = (1, 4)
        previous_hidden_state (np.ndarray):
            Context vector h_{t-1} carried over from previous timestep (h_0 = 0 at t=0).
            Shape: (B, H) = (1, 3)
        input_to_hidden_weights (np.ndarray):
            Projection matrix W_{xh} mapping inputs to recurrent neurons.
            Shape: (D, H) = (4, 3)
        hidden_to_hidden_weights (np.ndarray):
            Transition matrix W_{hh} mixing past memory into current step.
            Shape: (H, H) = (3, 3)
        hidden_bias (np.ndarray):
            Additive bias offset b_h for the hidden state.
            Shape: (1, H) = (1, 3)

    Returns:
        tuple (current_hidden_state, hidden_pre_activation):
            - current_hidden_state (np.ndarray): h_t in range (-1, 1). Shape: (B, H) = (1, 3)
            - hidden_pre_activation (np.ndarray): z_t before tanh. Shape: (B, H) = (1, 3)
    """
    # 1. Compute current input influence: (B, D) @ (D, H) -> (B, H)
    # When current_input is a one-hot row, this effectively selects one row of W_{xh}.
    input_contributions = current_input @ input_to_hidden_weights

    # 2. Compute previous memory influence: (B, H) @ (H, H) -> (B, H)
    # At t=0 with h_0 = zeros: [[0, 0, 0]] @ W_{hh} = [[0, 0, 0]]
    previous_memory_contribution = (
        previous_hidden_state @ hidden_to_hidden_weights
    )

    # 3. Combine linear influences and bias into raw memory z_t
    hidden_pre_activation = (
        input_contributions
        + previous_memory_contribution
        + hidden_bias
    )

    # 4. Bound memory values into range (-1, 1) using tanh activation
    current_hidden_state = np.tanh(hidden_pre_activation)

    return current_hidden_state, hidden_pre_activation


def stable_softmax(unnormalized_output_scores):
    """
    Converts raw unnormalized class scores (logits) into a valid categorical probability distribution.

    Mathematical Formulation:
        P(y = k | o) = exp(o_k - max_j(o_j)) / sum_j exp(o_j - max_j(o_j))

    Numerical Stability Rationale:
        Directly calculating exp(o) causes catastrophic floating-point overflow for large
        positive numbers (e.g., exp(1000) = inf, leading to inf / inf = NaN).
        Subtracting max(o) across each row shifts the maximum exponent argument to 0.0:
            exp(o_k - max) <= exp(0) = 1.0
        This mathematically preserves the exact probability distribution while guaranteeing
        zero risk of numerical overflow.

    Parameters:
        unnormalized_output_scores (np.ndarray):
            Raw output logits from the linear dense layer.
            Shape: (B, K) or (T, B, K) = (1, 4)

    Returns:
        probabilities (np.ndarray):
            Normalized probabilities summing to 1.0 across the last dimension.
            Shape: Same shape as input.
    """
    largest_score_per_row = np.max(
        unnormalized_output_scores,
        axis=1,
        keepdims=True,
    )

    shifted_output_scores = (
        unnormalized_output_scores - largest_score_per_row
    )

    exponetiated_scores = np.exp(shifted_output_scores)

    total_exponential_score_per_row = np.sum(
        exponetiated_scores,
        axis=1,
        keepdims=True,
    )

    probabilities = (
        exponetiated_scores / total_exponential_score_per_row
    )

    return probabilities


def rnn_output_forward(
    current_hidden_state,
    hidden_to_output_weights,
    output_bias,
):
    """
    Projects hidden recurrent memory to output vocabulary scores and normalized probabilities.

    Mathematical Formulation:
        1. Linear Projection:
           o_t = h_t @ W_{hy} + b_y
        2. Softmax Normalization:
           y_hat_t = softmax(o_t)

    Parameters:
        current_hidden_state (np.ndarray):
            Activated hidden memory state h_t at timestep t.
            Shape: (B, H) = (1, 3)
        hidden_to_output_weights (np.ndarray):
            Projection weight matrix W_{hy} mapping hidden state to output classes.
            Shape: (H, K) = (3, 4)
        output_bias (np.ndarray):
            Additive bias b_y for the output layer.
            Shape: (1, K) = (1, 4)

    Returns:
        tuple (unnormalized_output_scores, output_probabilities):
            - unnormalized_output_scores (np.ndarray): Raw class logits o_t. Shape: (B, K) = (1, 4)
            - output_probabilities (np.ndarray): Softmax probabilities y_hat_t. Shape: (B, K) = (1, 4)
    """
    unnormalized_output_scores = (
        current_hidden_state @ hidden_to_output_weights + output_bias
    )

    output_probabilities = stable_softmax(
        unnormalized_output_scores
    )

    return unnormalized_output_scores, output_probabilities


def rnn_forward(
    input_sequence,
    initial_hidden_state,
    input_to_hidden_weights,
    hidden_to_hidden_weights,
    hidden_bias,
    hidden_to_output_weights,
    output_bias,
):
    """
    Executes the complete unrolled forward pass across all sequential timesteps.

    Mathematical Flow Across Time (for t = 0, ..., T-1):
        x_t = input_sequence[:, t, :]
        h_t, z_t = rnn_cell_forward(x_t, h_{t-1}, W_{xh}, W_{hh}, b_h)
        o_t, y_hat_t = rnn_output_forward(h_t, W_{hy}, b_y)

    Unrolled Computational Graph:
        x_0 ('h')          x_1 ('e')          x_2 ('l')          x_3 ('l')
           │                  │                  │                  │
           ▼                  ▼                  ▼                  ▼
        ┌──────┐           ┌──────┐           ┌──────┐           ┌──────┐
        │ Cell │◄── h_0=0  │ Cell │◄── h_1    │ Cell │◄── h_2    │ Cell │◄── h_3
        └──────┘           └──────┘           └──────┘           └──────┘
           │                  │                  │                  │
           ├──► h_1           ├──► h_2           ├──► h_3           ├──► h_4
           ▼                  ▼                  ▼                  ▼
        ┌──────┐           ┌──────┐           ┌──────┐           ┌──────┐
        │ Head │           │ Head │           │ Head │           │ Head │
        └──────┘           └──────┘           └──────┘           └──────┘
           │                  │                  │                  │
           ▼                  ▼                  ▼                  ▼
        y_hat_0 ('e')      y_hat_1 ('l')      y_hat_2 ('l')      y_hat_3 ('o')

    Parameters:
        input_sequence (np.ndarray):
            Batched input sequence tensor. Shape: (B, T, D) = (1, 4, 4)
        initial_hidden_state (np.ndarray):
            Initial recurrent memory h_0 before processing sequence. Shape: (B, H) = (1, 3)
        input_to_hidden_weights (np.ndarray):
            Matrix W_{xh}. Shape: (D, H) = (4, 3)
        hidden_to_hidden_weights (np.ndarray):
            Matrix W_{hh}. Shape: (H, H) = (3, 3)
        hidden_bias (np.ndarray):
            Vector b_h. Shape: (1, H) = (1, 3)
        hidden_to_output_weights (np.ndarray):
            Matrix W_{hy}. Shape: (H, K) = (3, 4)
        output_bias (np.ndarray):
            Vector b_y. Shape: (1, K) = (1, 4)

    Returns:
        dict:
            - "all_hidden_states": (T+1, B, H) containing [h_0, h_1, ..., h_T]
            - "all_unnormalized_output_scores": (T, B, K) logits across all steps
            - "all_output_probabilities": (T, B, K) softmax probabilities across all steps
    """
    batch_size, number_of_timesteps, input_dimension = input_sequence.shape

    current_hidden_state = initial_hidden_state

    all_hidden_states = [initial_hidden_state]
    all_unnormalized_output_scores = []
    all_output_probabilities = []

    for timestep_index in range(number_of_timesteps):
        current_input = input_sequence[:, timestep_index, :]

        current_hidden_state, hidden_pre_activation = (
            rnn_cell_forward(
                current_input,
                current_hidden_state,
                input_to_hidden_weights,
                hidden_to_hidden_weights,
                hidden_bias,
            )
        )

        unnormalized_output_scores, output_probabilities = (
            rnn_output_forward(
                current_hidden_state,
                hidden_to_output_weights,
                output_bias,
            )
        )

        all_hidden_states.append(current_hidden_state)
        all_unnormalized_output_scores.append(unnormalized_output_scores)
        all_output_probabilities.append(output_probabilities)

    return {
        "all_hidden_states": np.stack(all_hidden_states, axis=0),
        "all_unnormalized_output_scores": np.stack(
            all_unnormalized_output_scores,
            axis=0,
        ),
        "all_output_probabilities": np.stack(
            all_output_probabilities,
            axis=0,
        ),
    }


# =============================================================================
# PART 2: SEQUENCE LOSS COMPUTATION
# =============================================================================

def sequence_cross_entropy(
    output_probabilities,
    target_sequence,
):
    """
    Computes Many-to-Many Categorical Cross-Entropy Loss summed across all timesteps.

    Mathematical Formulation:
        For each timestep t in 1, ..., T:
            L_t = - sum_{k=0}^{K-1} Y_{t, k} * ln(y_hat_{t, k} + epsilon)
        Total Sequence Loss:
            L_{seq} = sum_{t=1}^T L_t
        Average Loss per Timestep:
            L_{avg} = L_{seq} / T

    Parameters:
        output_probabilities (np.ndarray):
            Predicted probability distribution across all timesteps.
            Shape: (T, B, K) = (4, 1, 4)
        target_sequence (np.ndarray):
            Ground-truth one-hot encoded targets across all timesteps.
            Shape: (B, T, K) = (1, 4, 4) for "ello"

    Returns:
        tuple (total_sequence_loss, average_loss_per_timestep):
            - total_sequence_loss (float): Scalar sum of cross-entropy over all timesteps
            - average_loss_per_timestep (float): Scalar average loss per timestep
    """
    numerical_safety_value = 1e-12

    # Swap axes from (T, B, K) to (B, T, K) to align with target_sequence
    output_probabilities_by_batch = np.swapaxes(
        output_probabilities,
        0,
        1,
    )

    # Negative log likelihood: -ln(p + epsilon)
    negative_log_likelihoods = -np.log(
        output_probabilities_by_batch + numerical_safety_value
    )

    # Select loss for true one-hot class at each timestep: Y_{t, k} * -ln(p_{t, k})
    loss_for_each_class = (
        target_sequence * negative_log_likelihoods
    )

    total_sequence_loss = np.sum(loss_for_each_class)

    average_loss_per_timestep = (
        total_sequence_loss / target_sequence.shape[1]
    )

    return total_sequence_loss, average_loss_per_timestep


# =============================================================================
# PART 3: BACKPROPAGATION THROUGH TIME (BPTT) & PARAMETER GRADIENTS
# =============================================================================

def output_score_gradients(
    output_probabilities,
    target_sequence,
):
    """
    Calculates error at the output layer: derivative of cross-entropy w.r.t logits o_t.

    Mathematical Formulation:
        delta_{o, t} = dL / do_t = y_hat_t - Y_t

    Calculus Derivation:
        Combining the Softmax function (y_hat_k = exp(o_k) / sum_j exp(o_j))
        with Categorical Cross-Entropy (L = -sum_k Y_k * ln(y_hat_k)):
            Case 1 (k = target class, Y_k = 1):
                dL / do_k = y_hat_k - 1
            Case 2 (k != target class, Y_k = 0):
                dL / do_k = y_hat_k - 0
        Combined vector form across all classes:
            dL / do_t = y_hat_t - Y_t

    Parameters:
        output_probabilities (np.ndarray):
            Predicted probabilities. Shape: (T, B, K) = (4, 1, 4)
        target_sequence (np.ndarray):
            One-hot target characters. Shape: (B, T, K) = (1, 4, 4)

    Returns:
        output_score_errors (np.ndarray):
            Logit errors delta_{o, t}. Shape: (T, B, K) = (4, 1, 4)
    """
    target_sequence_by_timestep = np.swapaxes(
        target_sequence,
        0,
        1,
    )

    output_score_errors = (
        output_probabilities
        - target_sequence_by_timestep
    )

    return output_score_errors


def output_layer_parameter_gradients(
    all_hidden_states,
    output_score_errors,
):
    """
    Calculates parameter gradients for the output projection layer (W_{hy}, b_y).

    Mathematical Formulation:
        Because weights are shared across all timesteps, gradients accumulate over time:
            dL / dW_{hy} = sum_{t=1}^T (h_t)^T @ delta_{o, t}
            dL / db_y    = sum_{t=1}^T delta_{o, t}

    Dimensions Check:
        (h_t)^T is (H, B) = (3, 1)
        delta_{o, t} is (B, K) = (1, 4)
        Result: (3, 1) @ (1, 4) -> (3, 4), matching shape of W_{hy}.

    Parameters:
        all_hidden_states (np.ndarray):
            History of hidden states [h_0, h_1, ..., h_T]. Shape: (T+1, B, H) = (5, 1, 3)
        output_score_errors (np.ndarray):
            Logit errors delta_{o, t}. Shape: (T, B, K) = (4, 1, 4)

    Returns:
        tuple (hidden_to_output_weight_gradients, output_bias_gradients):
            - dL / dW_{hy}. Shape: (H, K) = (3, 4)
            - dL / db_y.    Shape: (1, K) = (1, 4)
    """
    hidden_state_size = all_hidden_states.shape[2]
    output_size = output_score_errors.shape[2]

    hidden_to_output_weight_gradients = np.zeros(
        (hidden_state_size, output_size)
    )

    output_bias_gradients = np.zeros(
        (1, output_size)
    )

    number_of_timesteps = output_score_errors.shape[0]

    for timestep_index in range(number_of_timesteps):
        # Note: all_hidden_states has h_0 at index 0, so h_1 is at index 1
        current_hidden_state = all_hidden_states[
            timestep_index + 1
        ]

        current_output_score_error = output_score_errors[
            timestep_index
        ]

        # Accumulate weight gradients: (H, B) @ (B, K) -> (H, K)
        hidden_to_output_weight_gradients += (
            current_hidden_state.T
            @ current_output_score_error
        )

        # Accumulate bias gradients: sum across batch dimension (1, K)
        output_bias_gradients += current_output_score_error

    return (
        hidden_to_output_weight_gradients,
        output_bias_gradients,
    )


def direct_hidden_state_gradients(
    output_score_errors,
    hidden_to_output_weights,
):
    """
    Backpropagates output prediction errors into the activated hidden memory h_t.

    Mathematical Formulation:
        delta_{h, t}^{direct} = delta_{o, t} @ (W_{hy})^T

    Dimensions Check:
        delta_{o, t} is (B, K) = (1, 4)
        (W_{hy})^T is (K, H) = (4, 3)
        Result: (1, 4) @ (4, 3) -> (1, 3), matching shape of h_t.

    Parameters:
        output_score_errors (np.ndarray):
            Logit errors delta_{o, t}. Shape: (T, B, K) = (4, 1, 4)
        hidden_to_output_weights (np.ndarray):
            Matrix W_{hy}. Shape: (H, K) = (3, 4)

    Returns:
        direct_hidden_state_errors (np.ndarray):
            Direct error on h_t before temporal recurrence. Shape: (T, B, H) = (4, 1, 3)
    """
    direct_hidden_state_errors = (
        output_score_errors
        @ hidden_to_output_weights.T
    )

    return direct_hidden_state_errors


def backpropagate_through_hidden_states(
    all_hidden_states,
    direct_hidden_state_errors,
    hidden_to_hidden_weights,
):
    """
    Executes Backpropagation Through Time (BPTT) backwards from step T down to step 1.

    Mathematical Recurrence (for timestep t = T, T-1, ..., 1):
        1. Combine Direct Output Error and Future Recurrent Error:
           delta_{h, t}^{total} = delta_{h, t}^{direct} + delta_{h, t}^{future}
           (At the final step t = T, future error is 0: delta_{h, T}^{future} = 0)

        2. Backpropagate Through tanh Activation:
           d/dz_t tanh(z_t) = 1 - tanh^2(z_t) = 1 - (h_t)^2
           delta_{z, t} = delta_{h, t}^{total} * (1 - (h_t)^2)   [element-wise product]

        3. Propagate Error Backwards to Previous Hidden State (t-1):
           delta_{h, t-1}^{future} = delta_{z, t} @ (W_{hh})^T

    Backward Error Flow Across Time:
        t=3 ('l') ◄─────── delta_{z, 4} @ W_{hh}^T ◄─────── t=4 ('l')
           │                                                   │
           ├──► delta_{h, 3}^{total}                           ├──► delta_{h, 4}^{total}
           ▼                                                   ▼
        * (1 - h_3^2)                                       * (1 - h_4^2)
           │                                                   │
           ▼                                                   ▼
        delta_{z, 3}                                        delta_{z, 4}

    Parameters:
        all_hidden_states (np.ndarray):
            History of hidden states [h_0, ..., h_T]. Shape: (T+1, B, H) = (5, 1, 3)
        direct_hidden_state_errors (np.ndarray):
            Direct errors delta_{h, t}^{direct}. Shape: (T, B, H) = (4, 1, 3)
        hidden_to_hidden_weights (np.ndarray):
            Recurrent matrix W_{hh}. Shape: (H, H) = (3, 3)

    Returns:
        tuple (total_hidden_state_errors, hidden_pre_activation_errors):
            - total_hidden_state_errors (np.ndarray): delta_{h, t}^{total}. Shape: (T, B, H) = (4, 1, 3)
            - hidden_pre_activation_errors (np.ndarray): delta_{z, t}. Shape: (T, B, H) = (4, 1, 3)
    """
    number_of_timesteps = direct_hidden_state_errors.shape[0]

    total_hidden_state_errors = np.zeros_like(
        direct_hidden_state_errors
    )

    hidden_pre_activation_errors = np.zeros_like(
        direct_hidden_state_errors
    )

    # Initial future error at the final timestep T is zero (no future steps exist)
    future_hidden_state_error = np.zeros_like(
        direct_hidden_state_errors[0]
    )

    # Step backwards in time: T-1, T-2, ..., 0 (corresponding to timesteps T, ..., 1)
    for timestep_index in range(
        number_of_timesteps - 1,
        -1,
        -1,
    ):
        current_hidden_state = all_hidden_states[
            timestep_index + 1
        ]

        direct_hidden_state_error = (
            direct_hidden_state_errors[timestep_index]
        )

        # 1. Total hidden state error: direct output error + incoming future error
        total_hidden_state_error = (
            direct_hidden_state_error
            + future_hidden_state_error
        )

        # 2. Derivative of tanh: d/dz tanh(z) = 1 - tanh(z)^2 = 1 - h^2
        tanh_derivative = 1.0 - current_hidden_state ** 2

        # 3. Pre-activation gradient delta_{z, t}
        hidden_pre_activation_error = (
            total_hidden_state_error * tanh_derivative
        )

        total_hidden_state_errors[timestep_index] = (
            total_hidden_state_error
        )

        hidden_pre_activation_errors[timestep_index] = (
            hidden_pre_activation_error
        )

        # 4. Compute error passed backwards to timestep t-1: delta_{z, t} @ W_{hh}^T
        future_hidden_state_error = (
            hidden_pre_activation_error
            @ hidden_to_hidden_weights.T
        )

    return (
        total_hidden_state_errors,
        hidden_pre_activation_errors,
    )


def hidden_layer_parameter_gradients(
    input_sequence,
    all_hidden_states,
    hidden_pre_activation_errors,
):
    """
    Calculates parameter gradients for the recurrent cell (W_{xh}, W_{hh}, b_h).

    Mathematical Formulation:
        Because weights are tied across all timesteps, gradients accumulate over time:
            dL / dW_{xh} = sum_{t=1}^T (x_t)^T @ delta_{z, t}
            dL / dW_{hh} = sum_{t=1}^T (h_{t-1})^T @ delta_{z, t}
            dL / db_h    = sum_{t=1}^T delta_{z, t}

    Dimensions Check:
        (x_t)^T is (D, B) = (4, 1), delta_{z, t} is (B, H) = (1, 3) -> (4, 3) for W_{xh}
        (h_{t-1})^T is (H, B) = (3, 1), delta_{z, t} is (B, H) = (1, 3) -> (3, 3) for W_{hh}
        delta_{z, t} is (1, 3) for b_h

    Parameters:
        input_sequence (np.ndarray):
            Input sequence x_t across time. Shape: (B, T, D) = (1, 4, 4)
        all_hidden_states (np.ndarray):
            History of hidden states [h_0, ..., h_T]. Shape: (T+1, B, H) = (5, 1, 3)
        hidden_pre_activation_errors (np.ndarray):
            BPTT pre-activation errors delta_{z, t}. Shape: (T, B, H) = (4, 1, 3)

    Returns:
        tuple (input_to_hidden_weight_gradients, hidden_to_hidden_weight_gradients, hidden_bias_gradients):
            - dL / dW_{xh}. Shape: (D, H) = (4, 3)
            - dL / dW_{hh}. Shape: (H, H) = (3, 3)
            - dL / db_h.    Shape: (1, H) = (1, 3)
    """
    input_size = input_sequence.shape[2]
    hidden_state_size = all_hidden_states.shape[2]

    input_to_hidden_weight_gradients = np.zeros(
        (input_size, hidden_state_size)
    )

    hidden_to_hidden_weight_gradients = np.zeros(
        (hidden_state_size, hidden_state_size)
    )

    hidden_bias_gradients = np.zeros(
        (1, hidden_state_size)
    )

    number_of_timesteps = input_sequence.shape[1]

    for timestep_index in range(number_of_timesteps):
        current_input = input_sequence[:, timestep_index, :]

        # Prior hidden state h_{t-1}: index 0 is h_0, index 1 is h_1, etc.
        previous_hidden_state = all_hidden_states[
            timestep_index
        ]

        current_hidden_pre_activation_error = (
            hidden_pre_activation_errors[timestep_index]
        )

        # Accumulate gradient for W_{xh}: (D, B) @ (B, H) -> (D, H)
        input_to_hidden_weight_gradients += (
            current_input.T
            @ current_hidden_pre_activation_error
        )

        # Accumulate gradient for W_{hh}: (H, B) @ (B, H) -> (H, H)
        hidden_to_hidden_weight_gradients += (
            previous_hidden_state.T
            @ current_hidden_pre_activation_error
        )

        # Accumulate gradient for b_h: sum over batch (1, H)
        hidden_bias_gradients += (
            current_hidden_pre_activation_error
        )

    return (
        input_to_hidden_weight_gradients,
        hidden_to_hidden_weight_gradients,
        hidden_bias_gradients,
    )


# =============================================================================
# PART 4: OPTIMIZATION & TRAINING
# =============================================================================

def clip_gradients_by_global_norm(
    parameter_gradients,
    maximum_gradient_norm,
):
    """
    Clips parameter gradients by global L2 norm to prevent exploding gradients.

    Mathematical Formulation:
        Global Gradient Norm:
            ||g|| = sqrt( sum_{p} sum_{i,j} (g_{p, i, j})^2 )
        Clipping Scale:
            scale = min(1.0, max_norm / (||g|| + 1e-12))
        Clipped Gradient for Parameter p:
            g_p^{clipped} = g_p * scale

    Why Gradient Clipping is Vital in RNNs:
        In recurrent backpropagation, repeated matrix multiplications by W_{hh}^T
        can cause gradients over long sequence steps to scale exponentially (exploding gradients).
        Global norm clipping rescales the overall gradient vector when ||g|| > max_norm,
        preserving the exact gradient trajectory direction while preventing catastrophic divergence.

    Parameters:
        parameter_gradients (dict):
            Mapping from parameter name string to its gradient array.
        maximum_gradient_norm (float):
            Upper threshold for the global L2 norm (e.g., 5.0).

    Returns:
        tuple (clipped_parameter_gradients, global_gradient_norm):
            - clipped_parameter_gradients (dict): Gradients scaled if norm > max_norm
            - global_gradient_norm (float): Unclipped scalar global L2 norm
    """
    total_squared_gradient_norm = sum(
        np.sum(gradient ** 2)
        for gradient in parameter_gradients.values()
    )

    global_gradient_norm = np.sqrt(
        total_squared_gradient_norm
    )

    clipping_scale = min(
        1.0,
        maximum_gradient_norm
        / (global_gradient_norm + 1e-12),
    )

    clipped_parameter_gradients = {
        parameter_name: gradient * clipping_scale
        for parameter_name, gradient in parameter_gradients.items()
    }

    return clipped_parameter_gradients, global_gradient_norm


def rnn_backward(
    input_sequence,
    target_sequence,
    forward_pass_results,
    model_parameters,
):
    """
    Assembles the complete backward pass: calculates gradients for all 5 parameters.

    Pipeline:
        1. Output Score Error: delta_{o, t} = y_hat_t - Y_t
        2. Output Parameters: dL / dW_{hy}, dL / db_y
        3. Direct Hidden Error: delta_{h, t}^{direct} = delta_{o, t} @ W_{hy}^T
        4. BPTT Recurrence: delta_{z, t} via backpropagation through time
        5. Recurrent Parameters: dL / dW_{xh}, dL / dW_{hh}, dL / db_h

    Parameters:
        input_sequence (np.ndarray): Input batch (B, T, D)
        target_sequence (np.ndarray): Target batch (B, T, K)
        forward_pass_results (dict): Caching from rnn_forward
        model_parameters (dict): Current parameter weights and biases

    Returns:
        dict: Mapping of parameter names to their accumulated gradients:
            - 'input_to_hidden_weights': (D, H)
            - 'hidden_to_hidden_weights': (H, H)
            - 'hidden_bias': (1, H)
            - 'hidden_to_output_weights': (H, K)
            - 'output_bias': (1, K)
    """
    output_probabilities = forward_pass_results[
        "all_output_probabilities"
    ]

    all_hidden_states = forward_pass_results[
        "all_hidden_states"
    ]

    # Step 1: Output errors delta_o
    output_score_errors = output_score_gradients(
        output_probabilities,
        target_sequence,
    )

    # Step 2: Gradients for W_{hy} and b_y
    (
        hidden_to_output_weight_gradients,
        output_bias_gradients,
    ) = output_layer_parameter_gradients(
        all_hidden_states,
        output_score_errors,
    )

    # Step 3: Direct error on h_t
    direct_hidden_state_errors = direct_hidden_state_gradients(
        output_score_errors,
        model_parameters["hidden_to_output_weights"],
    )

    # Step 4: Backpropagation through time (BPTT) for pre-activations delta_z
    (
        _,
        hidden_pre_activation_errors,
    ) = backpropagate_through_hidden_states(
        all_hidden_states,
        direct_hidden_state_errors,
        model_parameters["hidden_to_hidden_weights"],
    )

    # Step 5: Gradients for W_{xh}, W_{hh}, and b_h
    (
        input_to_hidden_weight_gradients,
        hidden_to_hidden_weight_gradients,
        hidden_bias_gradients,
    ) = hidden_layer_parameter_gradients(
        input_sequence,
        all_hidden_states,
        hidden_pre_activation_errors,
    )

    return {
        "input_to_hidden_weights": input_to_hidden_weight_gradients,
        "hidden_to_hidden_weights": hidden_to_hidden_weight_gradients,
        "hidden_bias": hidden_bias_gradients,
        "hidden_to_output_weights": hidden_to_output_weight_gradients,
        "output_bias": output_bias_gradients,
    }


def gradient_descent_update(
    model_parameters,
    parameter_gradients,
    learning_rate,
):
    """
    Updates model parameters using standard Gradient Descent (SGD).

    Mathematical Formulation:
        theta_{new} = theta_{old} - eta * dL / d(theta)

    Parameters:
        model_parameters (dict): Current weights and biases
        parameter_gradients (dict): Gradients (clipped)
        learning_rate (float): Learning rate step size eta

    Returns:
        updated_model_parameters (dict): New parameters after gradient step
    """
    updated_model_parameters = {
        parameter_name: (
            parameter_value
            - learning_rate
            * parameter_gradients[parameter_name]
        )
        for parameter_name, parameter_value in model_parameters.items()
    }

    return updated_model_parameters


def train_rnn(
    input_sequence,
    target_sequence,
    initial_hidden_state,
    model_parameters,
    learning_rate,
    number_of_epochs,
):
    """
    Runs repeated training iterations over the sequence to minimize cross-entropy loss.

    Training Loop (for each epoch):
        1. Forward pass: compute predictions and hidden states
        2. Sequence loss: compute cross-entropy
        3. Backward pass: derive parameter gradients via BPTT
        4. Clip gradients: scale by global norm if > 5.0
        5. Gradient descent: update all 5 parameters

    Parameters:
        input_sequence (np.ndarray): Input batch. Shape: (B, T, D)
        target_sequence (np.ndarray): Target batch. Shape: (B, T, K)
        initial_hidden_state (np.ndarray): Initial memory h_0. Shape: (B, H)
        model_parameters (dict): Initial parameter dictionary
        learning_rate (float): Step size eta (e.g., 0.1)
        number_of_epochs (int): Number of training iterations (e.g., 1000)

    Returns:
        tuple (current_model_parameters, loss_history):
            - current_model_parameters (dict): Trained model weights
            - loss_history (list): Sequence loss at each epoch
    """
    current_model_parameters = {
        parameter_name: parameter_value.copy()
        for parameter_name, parameter_value
        in model_parameters.items()
    }

    loss_history = []

    for epoch_index in range(number_of_epochs):
        # 1. Forward Pass
        forward_pass_results = rnn_forward(
            input_sequence,
            initial_hidden_state,
            current_model_parameters["input_to_hidden_weights"],
            current_model_parameters["hidden_to_hidden_weights"],
            current_model_parameters["hidden_bias"],
            current_model_parameters["hidden_to_output_weights"],
            current_model_parameters["output_bias"],
        )

        # 2. Compute Loss
        total_sequence_loss, _ = sequence_cross_entropy(
            forward_pass_results["all_output_probabilities"],
            target_sequence,
        )

        # 3. Backward Pass (BPTT)
        parameter_gradients = rnn_backward(
            input_sequence,
            target_sequence,
            forward_pass_results,
            current_model_parameters,
        )

        # 4. Clip Gradients by Global Norm
        clipped_parameter_gradients, _ = (
            clip_gradients_by_global_norm(
                parameter_gradients,
                maximum_gradient_norm=5.0,
            )
        )

        # 5. Parameter Update via Gradient Descent
        current_model_parameters = gradient_descent_update(
            current_model_parameters,
            clipped_parameter_gradients,
            learning_rate,
        )

        loss_history.append(total_sequence_loss)

        if epoch_index % 100 == 0:
            print(
                f"Epoch {epoch_index:4d} | "
                f"Loss: {total_sequence_loss:.6f}"
            )

    return current_model_parameters, loss_history


# =============================================================================
# PART 5: INFERENCE & AUTOREGRESSIVE TEXT GENERATION
# =============================================================================

def predict_next_characters(
    input_sequence,
    initial_hidden_state,
    trained_model_parameters,
    index_to_character,
):
    """
    Evaluates model predictions across the input sequence using argmax decoding.

    Parameters:
        input_sequence (np.ndarray): Test input sequence. Shape: (B, T, D)
        initial_hidden_state (np.ndarray): Initial memory h_0. Shape: (B, H)
        trained_model_parameters (dict): Trained model parameters
        index_to_character (dict): Vocabulary integer-to-character mapping

    Returns:
        tuple (predicted_characters, output_probabilities):
            - predicted_characters (list of str): Decoded character string list
            - output_probabilities (np.ndarray): Softmax probabilities across timesteps
    """
    forward_pass_results = rnn_forward(
        input_sequence,
        initial_hidden_state,
        trained_model_parameters["input_to_hidden_weights"],
        trained_model_parameters["hidden_to_hidden_weights"],
        trained_model_parameters["hidden_bias"],
        trained_model_parameters["hidden_to_output_weights"],
        trained_model_parameters["output_bias"],
    )

    output_probabilities = forward_pass_results[
        "all_output_probabilities"
    ]

    predicted_character_indices = np.argmax(
        output_probabilities[:, 0, :],
        axis=1,
    )

    predicted_characters = [
        index_to_character[character_index]
        for character_index in predicted_character_indices
    ]

    return predicted_characters, output_probabilities


def generate_characters(
    seed_character,
    number_of_characters_to_generate,
    initial_hidden_state,
    trained_model_parameters,
    character_to_index,
    index_to_character,
):
    """
    Autoregressively generates new text from an initial seed character.

    Autoregressive Feedback Loop:
        seed ('h') ──► [Step 1] ──► argmax ('e') ──┐
                                                   ├──► feeds as input x_t to next step
                       'e'      ──► [Step 2] ──► argmax ('l') ──┐
                                                                ▼
                                                 'l'      ──► [Step 3] ──► argmax ('l') ...

    Parameters:
        seed_character (str): The starting prompt character (e.g., 'h')
        number_of_characters_to_generate (int): Count of sequential steps to predict (e.g., 4)
        initial_hidden_state (np.ndarray): Initial recurrent memory state (1, H)
        trained_model_parameters (dict): Trained weights and biases
        character_to_index (dict): Vocabulary character-to-integer mapping
        index_to_character (dict): Vocabulary integer-to-character mapping

    Returns:
        generated_text (str): Complete generated text including seed (e.g., "hello")
    """
    vocabulary_size = len(character_to_index)

    current_hidden_state = initial_hidden_state.copy()

    current_input = np.zeros((1, vocabulary_size))
    current_input[0, character_to_index[seed_character]] = 1.0

    generated_characters = [seed_character]

    for _ in range(number_of_characters_to_generate):
        current_hidden_state, _ = rnn_cell_forward(
            current_input,
            current_hidden_state,
            trained_model_parameters["input_to_hidden_weights"],
            trained_model_parameters["hidden_to_hidden_weights"],
            trained_model_parameters["hidden_bias"],
        )

        _, output_probabilities = rnn_output_forward(
            current_hidden_state,
            trained_model_parameters["hidden_to_output_weights"],
            trained_model_parameters["output_bias"],
        )

        predicted_character_index = np.argmax(
            output_probabilities[0]
        )

        predicted_character = index_to_character[
            predicted_character_index
        ]

        generated_characters.append(predicted_character)

        # Prepare next input: one-hot encode the predicted character
        current_input = np.zeros((1, vocabulary_size))
        current_input[0, predicted_character_index] = 1.0

    return "".join(generated_characters)


# =============================================================================
# PART 6: MATHEMATICAL VERIFICATION (FINITE-DIFFERENCE GRADIENT CHECKING)
# =============================================================================

def calculate_total_sequence_loss(
    input_sequence,
    target_sequence,
    initial_hidden_state,
    model_parameters,
):
    """
    Helper function to compute total sequence loss for a given set of model parameters.

    Used by numerical_gradient_for_parameter_element to evaluate perturbed loss values L(theta +- eps).
    """
    forward_pass_results = rnn_forward(
        input_sequence,
        initial_hidden_state,
        model_parameters["input_to_hidden_weights"],
        model_parameters["hidden_to_hidden_weights"],
        model_parameters["hidden_bias"],
        model_parameters["hidden_to_output_weights"],
        model_parameters["output_bias"],
    )

    total_sequence_loss, _ = sequence_cross_entropy(
        forward_pass_results["all_output_probabilities"],
        target_sequence,
    )

    return total_sequence_loss


def numerical_gradient_for_parameter_element(
    input_sequence,
    target_sequence,
    initial_hidden_state,
    model_parameters,
    parameter_name,
    row_index,
    column_index,
    epsilon=1e-5,
):
    """
    Computes numerical gradient for a single weight using symmetric finite differences.

    Mathematical Formulation:
        dL / d theta_{i, j} ≈ [ L(theta_{i, j} + eps) - L(theta_{i, j} - eps) ] / (2 * eps)

    Verification Significance:
        Analytical BPTT involves intricate matrix transpositions, time unrolling,
        and activation derivatives. A symmetric finite difference provides an independent,
        ground-truth empirical derivative with approximation error O(eps^2).
        Matching analytical and numerical gradients confirms the BPTT calculus is 100% correct.

    Parameters:
        input_sequence (np.ndarray): Input batch (B, T, D)
        target_sequence (np.ndarray): Target batch (B, T, K)
        initial_hidden_state (np.ndarray): Initial memory (B, H)
        model_parameters (dict): Current parameters
        parameter_name (str): Key of parameter to test (e.g., 'input_to_hidden_weights')
        row_index (int): Row index i of weight element
        column_index (int): Column index j of weight element
        epsilon (float): Small numerical perturbation (default: 1e-5)

    Returns:
        numerical_gradient (float): Finite-difference gradient approximation
    """
    parameters_with_positive_change = {
        name: value.copy()
        for name, value in model_parameters.items()
    }

    parameters_with_negative_change = {
        name: value.copy()
        for name, value in model_parameters.items()
    }

    parameters_with_positive_change[
        parameter_name
    ][row_index, column_index] += epsilon

    parameters_with_negative_change[
        parameter_name
    ][row_index, column_index] -= epsilon

    loss_with_positive_change = calculate_total_sequence_loss(
        input_sequence,
        target_sequence,
        initial_hidden_state,
        parameters_with_positive_change,
    )

    loss_with_negative_change = calculate_total_sequence_loss(
        input_sequence,
        target_sequence,
        initial_hidden_state,
        parameters_with_negative_change,
    )

    return (
        loss_with_positive_change
        - loss_with_negative_change
    ) / (2.0 * epsilon)


# =============================================================================
# PART 7: EXECUTABLE EDUCATIONAL WALKTHROUGH & VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("==================================================================")
    print("   07 — Recurrent Neural Network (RNN) From Scratch Walkthrough   ")
    print("==================================================================")

    # -------------------------------------------------------------------------
    # Step 1: Initialize Deterministic Test Parameters & Inputs
    # -------------------------------------------------------------------------
    current_input = np.array([[1.0, 0.0, 0.0, 0.0]])         # 'h'
    previous_hidden_state = np.array([[0.0, 0.0, 0.0]])        # h_0 = [0, 0, 0]

    input_to_hidden_weights = np.array([
        [0.5, -0.2, 0.1],
        [-0.1, 0.4, -0.3],
        [0.2, -0.1, 0.5],
        [0.3, 0.2, -0.4],
    ])

    hidden_to_hidden_weights = np.array([
        [0.1, 0.3, -0.1],
        [-0.2, 0.1, 0.4],
        [0.3, -0.2, 0.2],
    ])

    hidden_bias = np.array([[0.1, -0.1, 0.0]])

    hidden_to_output_weights = np.array([
        [0.4, -0.3, 0.5, -0.1],
        [-0.2, 0.5, -0.1, 0.3],
        [0.1, -0.4, 0.2, 0.5],
    ])

    output_bias = np.array([[0.0, 0.1, -0.1, 0.2]])

    # -------------------------------------------------------------------------
    # Step 2: Single Step Recurrent Cell Forward Pass
    # -------------------------------------------------------------------------
    current_hidden_state, hidden_pre_activation = rnn_cell_forward(
        current_input,
        previous_hidden_state,
        input_to_hidden_weights,
        hidden_to_hidden_weights,
        hidden_bias,
    )

    # -------------------------------------------------------------------------
    # Step 3: Complete Forward Pass on "hell" -> "ello"
    # -------------------------------------------------------------------------
    input_sequence = np.array([
        [
            [1.0, 0.0, 0.0, 0.0],  # Step 0: 'h'
            [0.0, 1.0, 0.0, 0.0],  # Step 1: 'e'
            [0.0, 0.0, 1.0, 0.0],  # Step 2: 'l'
            [0.0, 0.0, 1.0, 0.0],  # Step 3: 'l'
        ]
    ])

    initial_hidden_state = np.zeros((1, 3))

    forward_pass_results = rnn_forward(
        input_sequence,
        initial_hidden_state,
        input_to_hidden_weights,
        hidden_to_hidden_weights,
        hidden_bias,
        hidden_to_output_weights,
        output_bias,
    )

    unnormalized_output_scores, output_probabilities = (
        rnn_output_forward(
            current_hidden_state,
            hidden_to_output_weights,
            output_bias,
        )
    )

    # Expected sequence targets: after 'h'->'e', after 'e'->'l', after 'l'->'l', after 'l'->'o'
    target_sequence = np.array([
        [
            [0.0, 1.0, 0.0, 0.0],  # Step 0: 'e'
            [0.0, 0.0, 1.0, 0.0],  # Step 1: 'l'
            [0.0, 0.0, 1.0, 0.0],  # Step 2: 'l'
            [0.0, 0.0, 0.0, 1.0],  # Step 3: 'o'
        ]
    ])

    # -------------------------------------------------------------------------
    # Step 4: Sequence Cross-Entropy Loss
    # -------------------------------------------------------------------------
    total_sequence_loss, average_loss_per_timestep = (
        sequence_cross_entropy(
            forward_pass_results["all_output_probabilities"],
            target_sequence,
        )
    )

    # -------------------------------------------------------------------------
    # Step 5: Output Gradients & Backward Pass (BPTT)
    # -------------------------------------------------------------------------
    output_score_errors = output_score_gradients(
        forward_pass_results["all_output_probabilities"],
        target_sequence,
    )

    (
        hidden_to_output_weight_gradients,
        output_bias_gradients,
    ) = output_layer_parameter_gradients(
        forward_pass_results["all_hidden_states"],
        output_score_errors,
    )

    direct_hidden_state_errors = (
        direct_hidden_state_gradients(
            output_score_errors,
            hidden_to_output_weights,
        )
    )

    (
        total_hidden_state_errors,
        hidden_pre_activation_errors,
    ) = backpropagate_through_hidden_states(
        forward_pass_results["all_hidden_states"],
        direct_hidden_state_errors,
        hidden_to_hidden_weights,
    )

    (
        input_to_hidden_weight_gradients,
        hidden_to_hidden_weight_gradients,
        hidden_bias_gradients,
    ) = hidden_layer_parameter_gradients(
        input_sequence,
        forward_pass_results["all_hidden_states"],
        hidden_pre_activation_errors,
    )

    model_parameters = {
        "input_to_hidden_weights": input_to_hidden_weights,
        "hidden_to_hidden_weights": hidden_to_hidden_weights,
        "hidden_bias": hidden_bias,
        "hidden_to_output_weights": hidden_to_output_weights,
        "output_bias": output_bias,
    }

    parameter_gradients = {
        "input_to_hidden_weights": input_to_hidden_weight_gradients,
        "hidden_to_hidden_weights": hidden_to_hidden_weight_gradients,
        "hidden_bias": hidden_bias_gradients,
        "hidden_to_output_weights": hidden_to_output_weight_gradients,
        "output_bias": output_bias_gradients,
    }

    # -------------------------------------------------------------------------
    # Step 6: Gradient Clipping & One Optimization Step
    # -------------------------------------------------------------------------
    clipped_parameter_gradients, global_gradient_norm = (
        clip_gradients_by_global_norm(
            parameter_gradients,
            maximum_gradient_norm=5.0,
        )
    )

    learning_rate = 0.01

    updated_model_parameters = gradient_descent_update(
        model_parameters,
        clipped_parameter_gradients,
        learning_rate,
    )

    updated_forward_pass_results = rnn_forward(
        input_sequence,
        initial_hidden_state,
        updated_model_parameters["input_to_hidden_weights"],
        updated_model_parameters["hidden_to_hidden_weights"],
        updated_model_parameters["hidden_bias"],
        updated_model_parameters["hidden_to_output_weights"],
        updated_model_parameters["output_bias"],
    )

    updated_total_sequence_loss, _ = sequence_cross_entropy(
        updated_forward_pass_results["all_output_probabilities"],
        target_sequence,
    )

    # -------------------------------------------------------------------------
    # Step 7: Train the Model for 1000 Epochs
    # -------------------------------------------------------------------------
    print("\n--- Training RNN on 'hello' Sequence ---")
    trained_model_parameters, loss_history = train_rnn(
        input_sequence,
        target_sequence,
        initial_hidden_state,
        model_parameters,
        learning_rate=0.1,
        number_of_epochs=1000,
    )

    # -------------------------------------------------------------------------
    # Step 8: Inference (Teacher-Forced Prediction & Text Generation)
    # -------------------------------------------------------------------------
    index_to_character = {
        0: "h",
        1: "e",
        2: "l",
        3: "o",
    }

    character_to_index = {
        "h": 0,
        "e": 1,
        "l": 2,
        "o": 3,
    }

    predicted_characters, trained_output_probabilities = (
        predict_next_characters(
            input_sequence,
            initial_hidden_state,
            trained_model_parameters,
            index_to_character,
        )
    )

    generated_text = generate_characters(
        seed_character="h",
        number_of_characters_to_generate=4,
        initial_hidden_state=initial_hidden_state,
        trained_model_parameters=trained_model_parameters,
        character_to_index=character_to_index,
        index_to_character=index_to_character,
    )

    # -------------------------------------------------------------------------
    # Step 9: Analytical vs Numerical Gradient Verification
    # -------------------------------------------------------------------------
    original_forward_pass_results = rnn_forward(
        input_sequence,
        initial_hidden_state,
        model_parameters["input_to_hidden_weights"],
        model_parameters["hidden_to_hidden_weights"],
        model_parameters["hidden_bias"],
        model_parameters["hidden_to_output_weights"],
        model_parameters["output_bias"],
    )

    analytical_parameter_gradients = rnn_backward(
        input_sequence,
        target_sequence,
        original_forward_pass_results,
        model_parameters,
    )

    numerical_gradient = numerical_gradient_for_parameter_element(
        input_sequence,
        target_sequence,
        initial_hidden_state,
        model_parameters,
        parameter_name="input_to_hidden_weights",
        row_index=0,
        column_index=0,
    )

    analytical_gradient = analytical_parameter_gradients[
        "input_to_hidden_weights"
    ][0, 0]

    # -------------------------------------------------------------------------
    # Step 10: Print Comprehensive Verification Summary
    # -------------------------------------------------------------------------
    print("\n--- Forward Pass Verification ---")
    print("Hidden pre-activation z_0:\n", hidden_pre_activation)
    print("Current hidden state h_0:\n", current_hidden_state)
    print("Unnormalized output scores o_0:\n", unnormalized_output_scores)
    print("Output probabilities y_hat_0:\n", output_probabilities)
    print("All hidden states across time:\n", forward_pass_results["all_hidden_states"])
    print("Total sequence loss before training:\n", total_sequence_loss)
    print("Average loss per timestep:\n", average_loss_per_timestep)

    print("\n--- Backward Pass (BPTT) Verification ---")
    print("Output score errors delta_o:\n", output_score_errors)
    print("Hidden-to-output weight gradients dL/dW_{hy}:\n", hidden_to_output_weight_gradients)
    print("Output bias gradients dL/db_y:\n", output_bias_gradients)
    print("Direct hidden-state errors delta_h^{direct}:\n", direct_hidden_state_errors)
    print("Hidden pre-activation errors (BPTT delta_z):\n", hidden_pre_activation_errors)
    print("Input-to-hidden weight gradients dL/dW_{xh}:\n", input_to_hidden_weight_gradients)
    print("Hidden-to-hidden weight gradients dL/dW_{hh}:\n", hidden_to_hidden_weight_gradients)
    print("Hidden bias gradients dL/db_h:\n", hidden_bias_gradients)
    print("Global gradient norm ||g||:\n", global_gradient_norm)
    print("Loss before update:", total_sequence_loss)
    print("Loss after one update:", updated_total_sequence_loss)

    print("\n--- Training & Text Generation Results ---")
    print("Initial training loss:", loss_history[0])
    print("Final training loss:  ", loss_history[-1])
    print("Predicted next characters for 'hell':", predicted_characters)
    print("Autoregressively generated text from 'h':", generated_text)

    print("\n--- Finite-Difference Gradient Check ---")
    print("Analytical BPTT gradient [W_{xh}[0, 0]]:", analytical_gradient)
    print("Numerical gradient       [W_{xh}[0, 0]]:", numerical_gradient)
    gradients_match = np.isclose(
        analytical_gradient,
        numerical_gradient,
        rtol=1e-4,
        atol=1e-6,
    )
    print("Gradients match within tolerance:", gradients_match)
    print("==================================================================\n")