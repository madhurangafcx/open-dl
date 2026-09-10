"""Vanilla character-level RNN from first principles using NumPy only.

This teaching implementation learns the next-character task ``hell -> ello``.
Shapes used throughout: B=batch size, T=timesteps, D=input/vocabulary size,
H=hidden-memory size, K=output/vocabulary size.

Forward equations at each timestep t:
    z_t = x_t @ W_xh + h_(t-1) @ W_hh + b_h
    h_t = tanh(z_t)
    o_t = h_t @ W_hy + b_y
    p_t = softmax(o_t)
"""

import numpy as np


# === Forward pass ===========================================================
def rnn_cell_forward(
    current_input,
    previous_hidden_state,
    input_to_hidden_weights,
    hidden_to_hidden_weights,
    hidden_bias,
):
    """Calculate one memory update.

    current_input: (B, D), previous_hidden_state: (B, H)
    input_to_hidden_weights/W_xh: (D, H), hidden_to_hidden_weights/W_hh: (H, H)
    hidden_bias/b_h: (1, H), broadcast once for each batch item.
    Returns h_t and z_t, both (B, H).
    """
    # x_t @ W_xh: contribution made by the current character.
    input_contribution = current_input @ input_to_hidden_weights
    # h_(t-1) @ W_hh: contribution made by memory of earlier characters.
    previous_memory_contribution = (
        previous_hidden_state @ hidden_to_hidden_weights
    )
    # z_t is the unrestricted, linear hidden value before the activation.
    hidden_pre_activation = (
        input_contribution + previous_memory_contribution + hidden_bias
    )
    # tanh maps each memory value into (-1, 1).
    current_hidden_state = np.tanh(hidden_pre_activation)
    return current_hidden_state, hidden_pre_activation


def stable_softmax(unnormalized_output_scores):
    """Return softmax probabilities for logits of shape (B, K).

    softmax(o_k) = exp(o_k) / sum_j exp(o_j).  Subtracting the greatest logit
    from every logit preserves the probability but prevents exp overflow.
    """
    largest_score_per_row = np.max(
        unnormalized_output_scores, axis=1, keepdims=True
    )
    shifted_output_scores = (
        unnormalized_output_scores - largest_score_per_row
    )
    exponentiated_scores = np.exp(shifted_output_scores)
    normalizing_total = np.sum(
        exponentiated_scores, axis=1, keepdims=True
    )
    return exponentiated_scores / normalizing_total


def rnn_output_forward(
    current_hidden_state,
    hidden_to_output_weights,
    output_bias,
):
    """Calculate o_t and p_t from h_t.

    h_t is (B, H), W_hy is (H, K), b_y is (1, K), and both returned
    arrays are (B, K).  Logits are scores; softmax turns them into probabilities.
    """
    unnormalized_output_scores = (
        current_hidden_state @ hidden_to_output_weights + output_bias
    )
    output_probabilities = stable_softmax(unnormalized_output_scores)
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
    """Unroll the one shared RNN cell across a sequence.

    input_sequence is (B, T, D).  The returned cache is time-major:
    hidden states (T+1, B, H), logits (T, B, K), probabilities (T, B, K).
    Keeping h_0 in the cache is essential because BPTT needs h_(t-1).
    """
    _, number_of_timesteps, _ = input_sequence.shape
    current_hidden_state = initial_hidden_state  # This is h_0 initially.
    all_hidden_states = [initial_hidden_state]
    all_unnormalized_output_scores = []
    all_output_probabilities = []

    for timestep_index in range(number_of_timesteps):
        # Slice x_t without discarding the batch dimension: (B, T, D) -> (B, D).
        current_input = input_sequence[:, timestep_index, :]
        current_hidden_state, _ = rnn_cell_forward(
            current_input, current_hidden_state, input_to_hidden_weights,
            hidden_to_hidden_weights, hidden_bias,
        )
        unnormalized_output_scores, output_probabilities = rnn_output_forward(
            current_hidden_state, hidden_to_output_weights, output_bias
        )
        # Cache every timestep because the backward pass needs all of them.
        all_hidden_states.append(current_hidden_state)
        all_unnormalized_output_scores.append(unnormalized_output_scores)
        all_output_probabilities.append(output_probabilities)

    return {
        "all_hidden_states": np.stack(all_hidden_states, axis=0),
        "all_unnormalized_output_scores": np.stack(
            all_unnormalized_output_scores, axis=0
        ),
        "all_output_probabilities": np.stack(all_output_probabilities, axis=0),
    }


# === Loss ===================================================================
def sequence_cross_entropy(output_probabilities, target_sequence):
    """Return total and average loss: L = -sum(y * log(p)).

    Predictions are time-major (T, B, K); targets follow input layout (B, T, K).
    A one-hot target removes all terms except the correct character's -log(p).
    """
    numerical_safety_value = 1e-12  # log(0) is undefined.
    output_probabilities_by_batch = np.swapaxes(output_probabilities, 0, 1)
    negative_log_likelihoods = -np.log(
        output_probabilities_by_batch + numerical_safety_value
    )
    loss_for_each_class = target_sequence * negative_log_likelihoods
    total_sequence_loss = np.sum(loss_for_each_class)
    average_loss_per_timestep = total_sequence_loss / target_sequence.shape[1]
    return total_sequence_loss, average_loss_per_timestep


def output_score_gradients(output_probabilities, target_sequence):
    """Return dL/do_t = p_t - y_t for softmax plus cross-entropy.

    This convenient simplification avoids manually differentiating softmax's
    K-by-K Jacobian.  Positive entries mean that class probability was too high.
    """
    target_sequence_by_timestep = np.swapaxes(target_sequence, 0, 1)
    return output_probabilities - target_sequence_by_timestep


# === Backpropagation Through Time ===========================================
def output_layer_parameter_gradients(all_hidden_states, output_score_errors):
    """Accumulate gradients for W_hy and b_y across shared timesteps.

    dW_hy += h_t.T @ do_t; db_y += do_t.
    """
    hidden_state_size = all_hidden_states.shape[2]
    output_size = output_score_errors.shape[2]
    hidden_to_output_weight_gradients = np.zeros((hidden_state_size, output_size))
    output_bias_gradients = np.zeros((1, output_size))

    for timestep_index in range(output_score_errors.shape[0]):
        # all_hidden_states[0] is h_0, so the t-th output uses index t+1.
        current_hidden_state = all_hidden_states[timestep_index + 1]
        current_output_score_error = output_score_errors[timestep_index]
        # (H, B) @ (B, K) gives (H, K), exactly the shape of W_hy.
        hidden_to_output_weight_gradients += (
            current_hidden_state.T @ current_output_score_error
        )
        # b_y was added directly, so its local gradient is do_t.
        output_bias_gradients += current_output_score_error

    return hidden_to_output_weight_gradients, output_bias_gradients


def direct_hidden_state_gradients(
    output_score_errors, hidden_to_output_weights
):
    """Return direct output-path error at each h_t: do_t @ W_hy.T.

    This omits the future-time path.  BPTT adds that path in the next function.
    """
    return output_score_errors @ hidden_to_output_weights.T


def backpropagate_through_hidden_states(
    all_hidden_states,
    direct_hidden_state_errors,
    hidden_to_hidden_weights,
):
    """Walk from T back to 1 and calculate total dL/dh_t and dL/dz_t.

    dL/dh_t = direct_output_path + future_recurrent_path
    dL/dz_t = dL/dh_t * (1 - h_t**2), because tanh'(z) = 1 - tanh(z)**2.
    """
    number_of_timesteps = direct_hidden_state_errors.shape[0]
    total_hidden_state_errors = np.zeros_like(direct_hidden_state_errors)
    hidden_pre_activation_errors = np.zeros_like(direct_hidden_state_errors)
    # The final state has no later timestep, therefore its future error is zero.
    future_hidden_state_error = np.zeros_like(direct_hidden_state_errors[0])

    # Reverse iteration is the "through time" part of BPTT.
    for timestep_index in range(number_of_timesteps - 1, -1, -1):
        current_hidden_state = all_hidden_states[timestep_index + 1]
        total_hidden_state_error = (
            direct_hidden_state_errors[timestep_index]
            + future_hidden_state_error
        )
        tanh_derivative = 1.0 - current_hidden_state ** 2
        hidden_pre_activation_error = (
            total_hidden_state_error * tanh_derivative
        )
        total_hidden_state_errors[timestep_index] = total_hidden_state_error
        hidden_pre_activation_errors[timestep_index] = hidden_pre_activation_error
        # Pass this timestep's error to h_(t-1) via W_hh.
        future_hidden_state_error = (
            hidden_pre_activation_error @ hidden_to_hidden_weights.T
        )

    return total_hidden_state_errors, hidden_pre_activation_errors


def hidden_layer_parameter_gradients(
    input_sequence,
    all_hidden_states,
    hidden_pre_activation_errors,
):
    """Accumulate recurrent parameter gradients across time.

    dW_xh += x_t.T @ dz_t; dW_hh += h_(t-1).T @ dz_t; db_h += dz_t.
    """
    input_size = input_sequence.shape[2]
    hidden_state_size = all_hidden_states.shape[2]
    input_to_hidden_weight_gradients = np.zeros((input_size, hidden_state_size))
    hidden_to_hidden_weight_gradients = np.zeros(
        (hidden_state_size, hidden_state_size)
    )
    hidden_bias_gradients = np.zeros((1, hidden_state_size))

    for timestep_index in range(input_sequence.shape[1]):
        current_input = input_sequence[:, timestep_index, :]
        previous_hidden_state = all_hidden_states[timestep_index]
        current_hidden_pre_activation_error = (
            hidden_pre_activation_errors[timestep_index]
        )
        # Each outer product has the shape of the parameter being differentiated.
        input_to_hidden_weight_gradients += (
            current_input.T @ current_hidden_pre_activation_error
        )
        hidden_to_hidden_weight_gradients += (
            previous_hidden_state.T @ current_hidden_pre_activation_error
        )
        hidden_bias_gradients += current_hidden_pre_activation_error

    return (
        input_to_hidden_weight_gradients,
        hidden_to_hidden_weight_gradients,
        hidden_bias_gradients,
    )


def rnn_backward(input_sequence, target_sequence, forward_pass_results, model_parameters):
    """Run all BPTT equations and return gradients for the five parameters."""
    output_probabilities = forward_pass_results["all_output_probabilities"]
    all_hidden_states = forward_pass_results["all_hidden_states"]
    output_score_errors = output_score_gradients(
        output_probabilities, target_sequence
    )
    hidden_to_output_weight_gradients, output_bias_gradients = (
        output_layer_parameter_gradients(all_hidden_states, output_score_errors)
    )
    direct_hidden_state_errors = direct_hidden_state_gradients(
        output_score_errors, model_parameters["hidden_to_output_weights"]
    )
    _, hidden_pre_activation_errors = backpropagate_through_hidden_states(
        all_hidden_states,
        direct_hidden_state_errors,
        model_parameters["hidden_to_hidden_weights"],
    )
    (
        input_to_hidden_weight_gradients,
        hidden_to_hidden_weight_gradients,
        hidden_bias_gradients,
    ) = hidden_layer_parameter_gradients(
        input_sequence, all_hidden_states, hidden_pre_activation_errors
    )
    return {
        "input_to_hidden_weights": input_to_hidden_weight_gradients,
        "hidden_to_hidden_weights": hidden_to_hidden_weight_gradients,
        "hidden_bias": hidden_bias_gradients,
        "hidden_to_output_weights": hidden_to_output_weight_gradients,
        "output_bias": output_bias_gradients,
    }


# === Optimisation ===========================================================
def clip_gradients_by_global_norm(parameter_gradients, maximum_gradient_norm):
    """Prevent exploding gradients by globally scaling them when necessary.

    g <- g * min(1, maximum_norm / ||g||_2).  Scaling every gradient by the
    same value retains the update direction.
    """
    total_squared_gradient_norm = sum(
        np.sum(gradient ** 2) for gradient in parameter_gradients.values()
    )
    global_gradient_norm = np.sqrt(total_squared_gradient_norm)
    clipping_scale = min(
        1.0, maximum_gradient_norm / (global_gradient_norm + 1e-12)
    )
    clipped_parameter_gradients = {
        name: gradient * clipping_scale
        for name, gradient in parameter_gradients.items()
    }
    return clipped_parameter_gradients, global_gradient_norm


def gradient_descent_update(model_parameters, parameter_gradients, learning_rate):
    """Return new theta values using theta <- theta - learning_rate*dL/dtheta."""
    # New arrays make it safe to retain and compare the old parameter dictionary.
    return {
        name: value - learning_rate * parameter_gradients[name]
        for name, value in model_parameters.items()
    }


def train_rnn(
    input_sequence, target_sequence, initial_hidden_state, model_parameters,
    learning_rate, number_of_epochs,
):
    """Repeat full-sequence forward pass, BPTT, clipping, and SGD updates."""
    # Never mutate the fixed teaching parameters used for mathematical checks.
    current_model_parameters = {
        name: value.copy() for name, value in model_parameters.items()
    }
    loss_history = []
    for epoch_index in range(number_of_epochs):
        forward_pass_results = rnn_forward(
            input_sequence, initial_hidden_state, **current_model_parameters
        )
        total_sequence_loss, _ = sequence_cross_entropy(
            forward_pass_results["all_output_probabilities"], target_sequence
        )
        parameter_gradients = rnn_backward(
            input_sequence, target_sequence, forward_pass_results,
            current_model_parameters,
        )
        clipped_parameter_gradients, _ = clip_gradients_by_global_norm(
            parameter_gradients, maximum_gradient_norm=5.0
        )
        current_model_parameters = gradient_descent_update(
            current_model_parameters, clipped_parameter_gradients, learning_rate
        )
        loss_history.append(total_sequence_loss)
        if epoch_index % 100 == 0:
            print(f"Epoch {epoch_index:4d} | Loss: {total_sequence_loss:.6f}")
    return current_model_parameters, loss_history


# === Inference ===============================================================
def predict_next_characters(
    input_sequence, initial_hidden_state, trained_model_parameters, index_to_character
):
    """Decode argmax predictions with teacher forcing (real inputs at all T)."""
    forward_pass_results = rnn_forward(
        input_sequence, initial_hidden_state, **trained_model_parameters
    )
    output_probabilities = forward_pass_results["all_output_probabilities"]
    # The tutorial has B=1, so batch element zero contains all predictions.
    predicted_character_indices = np.argmax(output_probabilities[:, 0, :], axis=1)
    predicted_characters = [
        index_to_character[index] for index in predicted_character_indices
    ]
    return predicted_characters, output_probabilities


def generate_characters(
    seed_character, number_of_characters_to_generate, initial_hidden_state,
    trained_model_parameters, character_to_index, index_to_character,
):
    """Generate autoregressively: each predicted character becomes next input."""
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
        # Greedy decoding selects the most likely class; it does not sample.
        predicted_character_index = np.argmax(output_probabilities[0])
        generated_characters.append(index_to_character[predicted_character_index])
        # One-hot encode the prediction so it becomes x_(t+1).
        current_input = np.zeros((1, vocabulary_size))
        current_input[0, predicted_character_index] = 1.0
    return "".join(generated_characters)


# === Numerical gradient verification ========================================
def calculate_total_sequence_loss(
    input_sequence, target_sequence, initial_hidden_state, model_parameters
):
    """Evaluate L(theta), used by the finite-difference gradient checker."""
    forward_pass_results = rnn_forward(
        input_sequence, initial_hidden_state, **model_parameters
    )
    total_sequence_loss, _ = sequence_cross_entropy(
        forward_pass_results["all_output_probabilities"], target_sequence
    )
    return total_sequence_loss


def numerical_gradient_for_parameter_element(
    input_sequence, target_sequence, initial_hidden_state, model_parameters,
    parameter_name, row_index, column_index, epsilon=1e-5,
):
    """Estimate one dL/dtheta with [L(theta+e)-L(theta-e)] / (2e).

    It is too slow for training, but agreement with BPTT is independent proof
    that the backward equations are correct.
    """
    parameters_with_positive_change = {
        name: value.copy() for name, value in model_parameters.items()
    }
    parameters_with_negative_change = {
        name: value.copy() for name, value in model_parameters.items()
    }
    parameters_with_positive_change[parameter_name][row_index, column_index] += epsilon
    parameters_with_negative_change[parameter_name][row_index, column_index] -= epsilon
    positive_loss = calculate_total_sequence_loss(
        input_sequence, target_sequence, initial_hidden_state,
        parameters_with_positive_change,
    )
    negative_loss = calculate_total_sequence_loss(
        input_sequence, target_sequence, initial_hidden_state,
        parameters_with_negative_change,
    )
    return (positive_loss - negative_loss) / (2.0 * epsilon)


if __name__ == "__main__":
    # Vocabulary index determines each one-hot vector's position.
    character_to_index = {"h": 0, "e": 1, "l": 2, "o": 3}
    index_to_character = {index: char for char, index in character_to_index.items()}

    # Fixed parameters match the step-by-step values in step-01-mathematics.md.
    model_parameters = {
        "input_to_hidden_weights": np.array([
            [0.5, -0.2, 0.1], [-0.1, 0.4, -0.3],
            [0.2, -0.1, 0.5], [0.3, 0.2, -0.4],
        ]),
        "hidden_to_hidden_weights": np.array([
            [0.1, 0.3, -0.1], [-0.2, 0.1, 0.4], [0.3, -0.2, 0.2],
        ]),
        "hidden_bias": np.array([[0.1, -0.1, 0.0]]),
        "hidden_to_output_weights": np.array([
            [0.4, -0.3, 0.5, -0.1], [-0.2, 0.5, -0.1, 0.3],
            [0.1, -0.4, 0.2, 0.5],
        ]),
        "output_bias": np.array([[0.0, 0.1, -0.1, 0.2]]),
    }
    # X is (B=1, T=4, D=4): h, e, l, l.
    input_sequence = np.array([[[1., 0., 0., 0.], [0., 1., 0., 0.],
                                [0., 0., 1., 0.], [0., 0., 1., 0.]]])
    # Y is (B=1, T=4, K=4): e, l, l, o.
    target_sequence = np.array([[[0., 1., 0., 0.], [0., 0., 1., 0.],
                                 [0., 0., 1., 0.], [0., 0., 0., 1.]]])
    initial_hidden_state = np.zeros((1, 3))  # h_0: no preceding memory.

    # Verify forward loss, backward gradients, and that one update lowers loss.
    forward_pass_results = rnn_forward(input_sequence, initial_hidden_state, **model_parameters)
    initial_loss, _ = sequence_cross_entropy(forward_pass_results["all_output_probabilities"], target_sequence)
    parameter_gradients = rnn_backward(input_sequence, target_sequence, forward_pass_results, model_parameters)
    clipped_gradients, global_gradient_norm = clip_gradients_by_global_norm(parameter_gradients, 5.0)
    updated_parameters = gradient_descent_update(model_parameters, clipped_gradients, 0.01)
    one_update_loss = calculate_total_sequence_loss(input_sequence, target_sequence, initial_hidden_state, updated_parameters)

    # Train the model, decode true-input predictions, and generate from only h.
    trained_parameters, loss_history = train_rnn(input_sequence, target_sequence, initial_hidden_state, model_parameters, 0.1, 1000)
    predicted_characters, _ = predict_next_characters(input_sequence, initial_hidden_state, trained_parameters, index_to_character)
    generated_text = generate_characters("h", 4, initial_hidden_state, trained_parameters, character_to_index, index_to_character)

    # Compare one BPTT derivative to an independently computed finite difference.
    analytical_gradient = parameter_gradients["input_to_hidden_weights"][0, 0]
    numerical_gradient = numerical_gradient_for_parameter_element(
        input_sequence, target_sequence, initial_hidden_state, model_parameters,
        "input_to_hidden_weights", 0, 0,
    )

    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Loss after one update: {one_update_loss:.6f}")
    print(f"Final training loss: {loss_history[-1]:.6f}")
    print(f"Global gradient norm: {global_gradient_norm:.6f}")
    print("Predicted next characters:", predicted_characters)
    print("Generated text:", generated_text)
    print(f"Analytical BPTT gradient: {analytical_gradient:.12f}")
    print(f"Numerical gradient: {numerical_gradient:.12f}")
    print("Gradients match:", np.isclose(analytical_gradient, numerical_gradient, rtol=1e-4, atol=1e-6))
