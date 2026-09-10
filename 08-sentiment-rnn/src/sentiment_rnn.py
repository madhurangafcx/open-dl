"""
08 — Many-to-One Recurrent Neural Network (RNN) for Sentiment Analysis
======================================================================

Architecture & NLP Overview:
----------------------------
This module implements an end-to-end word-level Many-to-One Recurrent Neural Network
(RNN) from first principles with pure NumPy for binary sentiment classification
(Positive = 1.0, Negative = 0.0).

Unlike character-level language models (which predict at every timestep), a Many-to-One
model reads an entire variable-length sentence, compresses its sequential meaning into
a final meaningful hidden state vector, and evaluates classification only once at the end.

Complete Natural Language Processing Pipeline:
----------------------------------------------
Raw Review Text ("The movie was REALLY good!")
       │
       ▼ 1. Lowercasing & Punctuation Stripping
Normalized Text ("the movie was really good")
       │
       ▼ 2. Whitespace Tokenization
Word Tokens (["the", "movie", "was", "really", "good"])
       │
       ▼ 3. Vocabulary Lookup
Integer Token IDs ([7, 4, 11, 8, 15])
       │
       ▼ 4. Post-Padding & True Length Tracking
Padded Sequence ([7, 4, 11, 8, 15, 0, 0]), True Length L = 5
       │
       ▼ 5. Embedding Matrix Lookup: e_t = E[w_t]
Continuous Word Embeddings Tensor (B, T, D) = (1, 7, 4)
       │
       ▼ 6. Padding-Safe Many-to-One RNN: h_t = tanh(e_t @ W_{xh} + h_{t-1} @ W_{hh} + b_h)
Sequential Hidden States, Freezing Memory on <PAD> Timesteps
       │
       ▼ 7. Final Meaningful Hidden State Extraction: h_L = h_5
Review Context Vector (B, H) = (1, 3)
       │
       ▼ 8. Linear Sentiment Projection: o = h_L @ W_{hy} + b_y
Raw Sentiment Logit (B, 1) = (1, 1)
       │
       ▼ 9. Numerically Stable Sigmoid: y_hat = 1 / (1 + exp(-o))
Predicted Positive Probability in range (0.0, 1.0)
       │
       ▼ 10. Binary Cross-Entropy Loss: L = -[y ln(y_hat) + (1-y) ln(1-y_hat)]
Optimization Scalar Loss

Mathematical Recurrence Equations:
----------------------------------
For each token w_t at timestep t in 1, ..., T:
    1. Word Embedding Vector:
       e_t = E[w_t] \in R^{1 x D}
    2. Recurrent Pre-activation:
       z_t = e_t @ W_{xh} + h_{t-1} @ W_{hh} + b_h \in R^{1 x H}
    3. Candidate Hidden Memory:
       h_t^{cand} = tanh(z_t) \in R^{1 x H}
    4. Padding-Safe Masking (Freeze state after sentence ends):
       h_t = h_t^{cand}  if t < L
       h_t = h_{t-1}     if t >= L (<PAD> token)
    5. Final Sentiment Classification (at true length L):
       o = h_L @ W_{hy} + b_y \in R^{1 x 1}
       y_hat = sigma(o) = 1 / (1 + exp(-o))
    6. Binary Cross-Entropy Loss:
       L = - [ y * ln(y_hat + epsilon) + (1 - y) * ln(1 - y_hat + epsilon) ]

Tensor Dimension Conventions:
-----------------------------
- B: Batch size                       (B = 1 in walkthrough, B = 8 in training batch)
- T: Maximum sequence length          (T = 6 in walkthrough, T = 4 in training batch)
- V: Vocabulary size                  (V = 16 educational tokens)
- D: Word embedding dimension         (D = 4 continuous latent features)
- H: Hidden recurrent state dimension (H = 3 internal memory neurons)
- K: Classification output dimension  (K = 1 binary sentiment scalar)

Learnable Parameter Budget (92 parameters total):
-------------------------------------------------
- E      (embedding_matrix):            shape (V, D) = (16, 4) -> 64 weights
- W_{xh} (embedding_to_hidden_weights): shape (D, H) = (4, 3)  -> 12 weights
- W_{hh} (hidden_to_hidden_weights):    shape (H, H) = (3, 3)  ->  9 weights
- b_h    (hidden_bias):                 shape (1, H) = (1, 3)  ->  3 biases
- W_{hy} (hidden_to_sentiment_weights): shape (H, 1) = (3, 1)  ->  3 weights
- b_y    (sentiment_output_bias):       shape (1, 1) = (1, 1)  ->  1 bias
"""

import re
import numpy as np


# =============================================================================
# PART 1: TEXT PREPROCESSING, TOKENIZATION & VOCABULARY
# =============================================================================

def normalize_and_tokenize(raw_text):
    """
    Cleans raw review text and splits it into a list of lowercase word tokens.

    Preprocessing Pipeline:
        "The movie was REALLY good!"
                 │ lowercasing
        "the movie was really good!"
                 │ remove punctuation via regex [^a-z\\s]
        "the movie was really good"
                 │ split on whitespace
        ["the", "movie", "was", "really", "good"]

    Parameters:
        raw_text (str): Arbitrary input string from user or dataset.

    Returns:
        word_tokens (list of str): Cleaned, lowercase individual word tokens.
    """
    lowercase_text = raw_text.lower()

    text_without_punctuation = re.sub(
        r"[^a-z\s]",
        "",
        lowercase_text,
    )

    word_tokens = text_without_punctuation.split()

    return word_tokens


def create_teaching_vocabulary():
    """
    Constructs the fixed educational vocabulary mapping words to unique integer token IDs.

    Vocabulary Structure (V = 16):
        Special Tokens:
            0: <PAD> - Padding placeholder for batch rectangularity
            1: <UNK> - Out-of-vocabulary fallback for unseen words
            2: <BOS> - Beginning of sequence marker
            3: <EOS> - End of sequence marker
        Sentiment Words:
            Positive: "great" (5), "good" (15)
            Negative: "terrible" (6), "bad" (9), "boring" (13)
            Negation: "not" (14)
            Context / Neutral: "movie" (4), "the" (7), "really" (8), "acting" (10), "was" (11), "film" (12)

    Returns:
        word_to_index (dict): Mapping from word string to unique integer index in [0, V-1].
    """
    word_to_index = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3,
        "movie": 4,
        "great": 5,
        "terrible": 6,
        "the": 7,
        "really": 8,
        "bad": 9,
        "acting": 10,
        "was": 11,
        "film": 12,
        "boring": 13,
        "not": 14,
        "good": 15,
    }

    return word_to_index


def convert_tokens_to_token_ids(
    word_tokens,
    word_to_index,
):
    """
    Converts a list of string tokens into a list of integer token IDs using the vocabulary.

    Out-of-Vocabulary (OOV) Handling:
        Any word not present in word_to_index is automatically assigned the index
        of the special unknown token '<UNK>' (index 1).

    Parameters:
        word_tokens (list of str): Tokenized words from normalize_and_tokenize.
        word_to_index (dict): Vocabulary lookup dictionary.

    Returns:
        token_ids (list of int): Sequential integer token identifiers.
    """
    unknown_token_index = word_to_index["<UNK>"]

    token_ids = [
        word_to_index.get(
            word_token,
            unknown_token_index,
        )
        for word_token in word_tokens
    ]

    return token_ids


def pad_or_truncate_token_ids(
    token_ids,
    maximum_sequence_length,
    padding_token_index,
):
    """
    Adjusts a token ID sequence to a uniform length while recording its true sentence length.

    Batch Padding Rationale:
        Tensors in NumPy require uniform rectangular dimensions across all samples in a batch.
        However, sentences naturally vary in word count. We post-pad shorter sentences with
        padding_token_index (0) up to maximum_sequence_length and return actual_sequence_length
        so that the recurrent network knows precisely when real sentence words end.

    Parameters:
        token_ids (list of int): Raw sequence of token IDs.
        maximum_sequence_length (int): Fixed target width T for the batch tensor.
        padding_token_index (int): ID of the padding token (<PAD> = 0).

    Returns:
        tuple (padded_token_ids, actual_sequence_length):
            - padded_token_ids (list of int): List of length maximum_sequence_length.
            - actual_sequence_length (int): True count of real words L <= maximum_sequence_length.
    """
    actual_sequence_length = min(
        len(token_ids),
        maximum_sequence_length,
    )

    truncated_token_ids = token_ids[
        :maximum_sequence_length
    ]

    number_of_padding_tokens = (
        maximum_sequence_length
        - len(truncated_token_ids)
    )

    padded_token_ids = (
        truncated_token_ids
        + [padding_token_index] * number_of_padding_tokens
    )

    return padded_token_ids, actual_sequence_length


def prepare_sentiment_batch(
    review_texts,
    word_to_index,
):
    """
    Preprocesses a collection of raw review sentences into a batched token ID tensor.

    Pipeline:
        1. Tokenizes each review string.
        2. Converts tokens to integer IDs.
        3. Identifies maximum sequence length T across the batch.
        4. Post-pads all reviews with <PAD> to width T.
        5. Packages sequences into 2D NumPy arrays.

    Parameters:
        review_texts (list of str): Raw text reviews in the batch.
        word_to_index (dict): Vocabulary lookup dictionary.

    Returns:
        tuple (token_id_batch, actual_sequence_lengths):
            - token_id_batch (np.ndarray): Padded integer matrix. Shape: (B, T)
            - actual_sequence_lengths (np.ndarray): 1D array of true lengths L. Shape: (B,)
    """
    padding_token_index = word_to_index["<PAD>"]

    token_id_sequences = []

    for review_text in review_texts:
        word_tokens = normalize_and_tokenize(
            review_text
        )

        token_ids = convert_tokens_to_token_ids(
            word_tokens,
            word_to_index,
        )

        token_id_sequences.append(token_ids)

    maximum_sequence_length = max(
        len(token_ids)
        for token_ids in token_id_sequences
    )

    padded_token_id_sequences = []
    actual_sequence_lengths = []

    for token_ids in token_id_sequences:
        padded_token_ids, actual_sequence_length = (
            pad_or_truncate_token_ids(
                token_ids,
                maximum_sequence_length,
                padding_token_index,
            )
        )

        padded_token_id_sequences.append(
            padded_token_ids
        )

        actual_sequence_lengths.append(
            actual_sequence_length
        )

    token_id_batch = np.array(
        padded_token_id_sequences,
        dtype=np.int64,
    )

    actual_sequence_lengths = np.array(
        actual_sequence_lengths,
        dtype=np.int64,
    )

    return token_id_batch, actual_sequence_lengths


# =============================================================================
# PART 2: CONTINUOUS WORD EMBEDDINGS
# =============================================================================

def create_teaching_embedding_matrix():
    """
    Instantiates the learnable dense word embedding matrix E of shape (V, D) = (16, 4).

    Embedding Geometry Rationale:
        Instead of high-dimensional, sparse, orthogonal one-hot vectors where all words
        are equidistant (||x_i - x_j|| = sqrt(2)), dense continuous embeddings place words
        into a smooth D-dimensional semantic vector space.
        Words with similar emotional polarity can cluster together, enabling the model
        to generalize across synonyms.

    Initial Weights Setup:
        - <PAD> (index 0): strictly initialized to [0, 0, 0, 0].
        - Semantically positive words ("good") and negative words ("not") receive distinct values.

    Returns:
        embedding_matrix (np.ndarray): Embedding lookup table E. Shape: (V, D) = (16, 4)
    """
    vocabulary_size = 16
    embedding_dimension = 4

    embedding_matrix = np.zeros(
        (vocabulary_size, embedding_dimension)
    )

    # Initial semantic embeddings for teaching walkthrough
    embedding_matrix[1] = [0.1, -0.1, 0.2, -0.2]     # <UNK>
    embedding_matrix[10] = [0.3, -0.2, 0.4, 0.1]     # acting
    embedding_matrix[11] = [0.1, 0.05, -0.1, 0.15]   # was
    embedding_matrix[14] = [-0.5, -0.3, 0.1, -0.6]   # not
    embedding_matrix[15] = [0.6, 0.5, -0.2, 0.4]     # good

    return embedding_matrix


def lookup_word_embeddings(
    token_id_batch,
    embedding_matrix,
):
    """
    Extracts continuous D-dimensional feature vectors for each token ID in the batch.

    Mathematical Formulation:
        For batch item b and timestep t:
            e_{b, t} = E[token_id_{b, t}] \in R^{1 x D}

    Parameters:
        token_id_batch (np.ndarray):
            Integer matrix of token IDs. Shape: (B, T)
        embedding_matrix (np.ndarray):
            Table E containing word vectors. Shape: (V, D) = (16, 4)

    Returns:
        word_embedding_batch (np.ndarray):
            Dense word vectors for each token. Shape: (B, T, D)
    """
    word_embedding_batch = embedding_matrix[token_id_batch]

    return word_embedding_batch


# =============================================================================
# PART 3: RECURRENT CELL & PADDING-SAFE FORWARD PASS
# =============================================================================

def create_teaching_recurrent_parameters():
    """
    Initializes recurrent layer weights and biases for the Many-to-One sentiment model.

    Parameter Shapes:
        - W_{xh} (embedding_to_hidden_weights): (D, H) = (4, 3)
        - W_{hh} (hidden_to_hidden_weights):    (H, H) = (3, 3)
        - b_h    (hidden_bias):                 (1, H) = (1, 3)

    Returns:
        tuple (embedding_to_hidden_weights, hidden_to_hidden_weights, hidden_bias)
    """
    embedding_to_hidden_weights = np.array([
        [0.4, -0.2, 0.3],
        [-0.3, 0.5, -0.1],
        [0.2, -0.1, 0.4],
        [0.1, 0.3, -0.2],
    ])

    hidden_to_hidden_weights = np.array([
        [0.2, 0.3, -0.1],
        [-0.1, 0.2, 0.4],
        [0.3, -0.2, 0.1],
    ])

    hidden_bias = np.array([[0.1, -0.05, 0.0]])

    return (
        embedding_to_hidden_weights,
        hidden_to_hidden_weights,
        hidden_bias,
    )


def sentiment_rnn_cell_forward(
    current_word_embedding,
    previous_hidden_state,
    embedding_to_hidden_weights,
    hidden_to_hidden_weights,
    hidden_bias,
):
    """
    Executes one recurrent step combining a single word embedding with prior memory.

    Mathematical Formulation:
        1. Input Projection:
           z_{input, t} = e_t @ W_{xh}
        2. Recurrent Memory Projection:
           z_{recurrent, t} = h_{t-1} @ W_{hh}
        3. Pre-activation:
           z_t = z_{input, t} + z_{recurrent, t} + b_h
        4. Non-linear Activation:
           h_t = tanh(z_t)

    Shapes:
        - current_word_embedding:     (B, D) = (1, 4)
        - previous_hidden_state:      (B, H) = (1, 3)
        - embedding_to_hidden_weights: (D, H) = (4, 3)
        - hidden_to_hidden_weights:   (H, H) = (3, 3)
        - hidden_bias:                (1, H) = (1, 3)
        ---------------------------------------------
        Returns:
        - current_hidden_state:       (B, H) = (1, 3)
        - hidden_pre_activation:      (B, H) = (1, 3)
    """
    embedding_contribution = (
        current_word_embedding
        @ embedding_to_hidden_weights
    )

    previous_memory_contribution = (
        previous_hidden_state
        @ hidden_to_hidden_weights
    )

    hidden_pre_activation = (
        embedding_contribution
        + previous_memory_contribution
        + hidden_bias
    )

    current_hidden_state = np.tanh(
        hidden_pre_activation
    )

    return current_hidden_state, hidden_pre_activation


def many_to_one_rnn_forward(
    word_embedding_batch,
    actual_sequence_lengths,
    initial_hidden_state,
    embedding_to_hidden_weights,
    hidden_to_hidden_weights,
    hidden_bias,
):
    """
    Unrolls the recurrent network across the sequence with strict padding masking.

    The Critical Masking Rule (Why We Must Freeze Padding):
        Even if <PAD> has a zero embedding vector (e_{pad} = [0, 0, 0, 0]), computing
        an unmasked recurrent step on padding would evaluate:
            h_t = tanh([0, 0, 0, 0] @ W_{xh} + h_{t-1} @ W_{hh} + b_h)
                = tanh(h_{t-1} @ W_{hh} + b_h) != h_{t-1}
        The hidden bias b_h and transition matrix W_{hh} would distort and corrupt
        the accumulated sentence meaning!
        To solve this, on padded timesteps (t >= L), the network strictly freezes the state:
            h_t = h_{t-1}

    Parameters:
        word_embedding_batch (np.ndarray):
            Embeddings for the entire sequence. Shape: (B, T, D)
        actual_sequence_lengths (np.ndarray):
            True word counts L for each batch item. Shape: (B,)
        initial_hidden_state (np.ndarray):
            Starting memory h_0 (typically all zeros). Shape: (B, H)
        embedding_to_hidden_weights (np.ndarray): W_{xh}. Shape: (D, H)
        hidden_to_hidden_weights (np.ndarray): W_{hh}. Shape: (H, H)
        hidden_bias (np.ndarray): b_h. Shape: (1, H)

    Returns:
        tuple (all_hidden_states, final_hidden_states):
            - all_hidden_states (np.ndarray): History [h_0, h_1, ..., h_T]. Shape: (B, T+1, H)
            - final_hidden_states (np.ndarray): The meaningful state h_L at actual length L. Shape: (B, H)
    """
    batch_size, maximum_sequence_length, _ = (
        word_embedding_batch.shape
    )

    current_hidden_state = initial_hidden_state

    # Store all hidden states across time, beginning with initial h_0 at index 0
    all_hidden_states = [initial_hidden_state]

    for timestep_index in range(
        maximum_sequence_length
    ):
        current_word_embedding = word_embedding_batch[
            :,
            timestep_index,
            :,
        ]

        candidate_hidden_state, _ = (
            sentiment_rnn_cell_forward(
                current_word_embedding,
                current_hidden_state,
                embedding_to_hidden_weights,
                hidden_to_hidden_weights,
                hidden_bias,
            )
        )

        # Boolean mask: True if timestep_index < actual sentence length L
        is_real_word_timestep = (
            timestep_index < actual_sequence_lengths
        )[:, np.newaxis]

        # Freeze hidden state if current token is <PAD>: h_t = h_{t-1}
        current_hidden_state = np.where(
            is_real_word_timestep,
            candidate_hidden_state,
            current_hidden_state,
        )

        all_hidden_states.append(current_hidden_state)

    # Stack into tensor of shape (B, T+1, H)
    all_hidden_states = np.stack(
        all_hidden_states,
        axis=1,
    )

    # Extract the exact hidden state h_L at the true sentence boundary L for each batch item
    batch_indices = np.arange(batch_size)

    final_hidden_states = all_hidden_states[
        batch_indices,
        actual_sequence_lengths,
    ]

    return all_hidden_states, final_hidden_states


# =============================================================================
# PART 4: BINARY CLASSIFICATION HEAD & LOSS
# =============================================================================

def stable_sigmoid(sentiment_logits):
    """
    Computes numerically stable Sigmoid activation mapping logits to probabilities in (0, 1).

    Mathematical Formulation:
        sigma(o) = 1 / (1 + exp(-o))

    Numerical Stability Rationale:
        When o is a large negative number (e.g., -1000), exp(-o) = exp(1000) causes overflow.
        We branch computation based on the sign of o:
            For o >= 0: sigma(o) = 1 / (1 + exp(-o))          (exponent argument <= 0)
            For o < 0:  sigma(o) = exp(o) / (1 + exp(o))      (exponent argument < 0)
        This completely eliminates overflow across all numerical ranges.

    Parameters:
        sentiment_logits (np.ndarray): Unnormalized class score o. Shape: (B, 1)

    Returns:
        sentiment_probabilities (np.ndarray): Values in (0.0, 1.0). Shape: (B, 1)
    """
    positive_logit_mask = sentiment_logits >= 0.0

    sentiment_probabilities = np.empty_like(
        sentiment_logits,
        dtype=np.float64,
    )

    # Branch 1: o >= 0
    sentiment_probabilities[positive_logit_mask] = (
        1.0
        / (
            1.0
            + np.exp(
                -sentiment_logits[positive_logit_mask]
            )
        )
    )

    # Branch 2: o < 0
    negative_logits = sentiment_logits[
        ~positive_logit_mask
    ]

    exponentiated_negative_logits = np.exp(
        negative_logits
    )

    sentiment_probabilities[~positive_logit_mask] = (
        exponentiated_negative_logits
        / (1.0 + exponentiated_negative_logits)
    )

    return sentiment_probabilities


def sentiment_output_forward(
    final_hidden_states,
    hidden_to_sentiment_weights,
    sentiment_output_bias,
):
    """
    Projects the final sentence hidden state h_L to a single binary sentiment probability.

    Mathematical Formulation:
        1. Linear Projection (Dense Head):
           o = h_L @ W_{hy} + b_y \in R^{B x 1}
        2. Sigmoid Activation:
           y_hat = sigma(o) \in R^{B x 1}

    Parameters:
        final_hidden_states (np.ndarray):
            Sentence context vector h_L at true length L. Shape: (B, H) = (1, 3)
        hidden_to_sentiment_weights (np.ndarray):
            Projection matrix W_{hy}. Shape: (H, 1) = (3, 1)
        sentiment_output_bias (np.ndarray):
            Additive output bias b_y. Shape: (1, 1)

    Returns:
        tuple (sentiment_logits, positive_sentiment_probabilities):
            - sentiment_logits (np.ndarray): Raw logit o. Shape: (B, 1)
            - positive_sentiment_probabilities (np.ndarray): Probability y_hat. Shape: (B, 1)
    """
    sentiment_logits = (
        final_hidden_states
        @ hidden_to_sentiment_weights
        + sentiment_output_bias
    )

    positive_sentiment_probabilities = stable_sigmoid(
        sentiment_logits
    )

    return (
        sentiment_logits,
        positive_sentiment_probabilities,
    )


def binary_cross_entropy(
    positive_sentiment_probabilities,
    sentiment_labels,
):
    """
    Evaluates Binary Cross-Entropy Loss over the mini-batch of reviews.

    Mathematical Formulation:
        For target label y in {0.0 (Negative), 1.0 (Positive)}:
            L_i = - [ y_i * ln(y_hat_i + eps) + (1 - y_i) * ln(1 - y_hat_i + eps) ]
        Average Batch Loss:
            L = (1 / B) * sum_{i=1}^B L_i

    Parameters:
        positive_sentiment_probabilities (np.ndarray):
            Predicted probabilities y_hat. Shape: (B, 1)
        sentiment_labels (np.ndarray):
            Ground truth binary targets y. Shape: (B, 1)

    Returns:
        average_binary_cross_entropy_loss (float): Scalar mean loss over batch.
    """
    numerical_safety_value = 1e-12

    safe_probabilities = np.clip(
        positive_sentiment_probabilities,
        numerical_safety_value,
        1.0 - numerical_safety_value,
    )

    loss_for_each_review = -(
        sentiment_labels
        * np.log(safe_probabilities)
        + (1.0 - sentiment_labels)
        * np.log(1.0 - safe_probabilities)
    )

    average_binary_cross_entropy_loss = np.mean(
        loss_for_each_review
    )

    return average_binary_cross_entropy_loss


def sentiment_rnn_forward(
    token_id_batch,
    actual_sequence_lengths,
    initial_hidden_state,
    model_parameters,
):
    """
    Executes the full forward pass: Embeddings -> Many-to-One RNN -> Sentiment Head.

    Parameters:
        token_id_batch (np.ndarray): Padded token indices. Shape: (B, T)
        actual_sequence_lengths (np.ndarray): True sentence lengths L. Shape: (B,)
        initial_hidden_state (np.ndarray): Initial memory h_0. Shape: (B, H)
        model_parameters (dict): Dictionary containing all 6 learnable parameters:
            - 'embedding_matrix': (V, D)
            - 'embedding_to_hidden_weights': (D, H)
            - 'hidden_to_hidden_weights': (H, H)
            - 'hidden_bias': (1, H)
            - 'hidden_to_sentiment_weights': (H, 1)
            - 'sentiment_output_bias': (1, 1)

    Returns:
        dict: Intermediate tensors required for inference and BPTT backpropagation.
    """
    word_embedding_batch = lookup_word_embeddings(
        token_id_batch,
        model_parameters["embedding_matrix"],
    )

    all_hidden_states, final_hidden_states = (
        many_to_one_rnn_forward(
            word_embedding_batch,
            actual_sequence_lengths,
            initial_hidden_state,
            model_parameters["embedding_to_hidden_weights"],
            model_parameters["hidden_to_hidden_weights"],
            model_parameters["hidden_bias"],
        )
    )

    (
        sentiment_logits,
        positive_sentiment_probabilities,
    ) = sentiment_output_forward(
        final_hidden_states,
        model_parameters["hidden_to_sentiment_weights"],
        model_parameters["sentiment_output_bias"],
    )

    return {
        "word_embedding_batch": word_embedding_batch,
        "all_hidden_states": all_hidden_states,
        "final_hidden_states": final_hidden_states,
        "sentiment_logits": sentiment_logits,
        "positive_sentiment_probabilities": (
            positive_sentiment_probabilities
        ),
    }


# =============================================================================
# PART 5: BACKPROPAGATION THROUGH TIME (BPTT) & EMBEDDING GRADIENTS
# =============================================================================

def sentiment_output_backward(
    final_hidden_states,
    positive_sentiment_probabilities,
    sentiment_labels,
    hidden_to_sentiment_weights,
):
    """
    Computes output layer gradients and propagates error into the final hidden state h_L.

    Mathematical Formulation:
        1. Derivative of Binary Cross-Entropy w.r.t Sigmoid Logit o:
           delta_o = dL / do = (y_hat - y) / B \in R^{B x 1}
        2. Gradient for Output Projection Weights W_{hy}:
           dL / dW_{hy} = (h_L)^T @ delta_o \in R^{H x 1}
        3. Gradient for Output Bias b_y:
           dL / db_y = sum_{b=1}^B delta_{o, b} \in R^{1 x 1}
        4. Error Injected into Final Meaningful Hidden State h_L:
           delta_{h, L}^{final} = delta_o @ (W_{hy})^T \in R^{B x H}

    Parameters:
        final_hidden_states (np.ndarray): h_L. Shape: (B, H)
        positive_sentiment_probabilities (np.ndarray): y_hat. Shape: (B, 1)
        sentiment_labels (np.ndarray): y. Shape: (B, 1)
        hidden_to_sentiment_weights (np.ndarray): W_{hy}. Shape: (H, 1)

    Returns:
        tuple (sentiment_logit_errors, hidden_to_sentiment_weight_gradients,
               sentiment_output_bias_gradients, final_hidden_state_errors)
    """
    batch_size = sentiment_labels.shape[0]

    sentiment_logit_errors = (
        positive_sentiment_probabilities
        - sentiment_labels
    ) / batch_size

    hidden_to_sentiment_weight_gradients = (
        final_hidden_states.T
        @ sentiment_logit_errors
    )

    sentiment_output_bias_gradients = np.sum(
        sentiment_logit_errors,
        axis=0,
        keepdims=True,
    )

    final_hidden_state_errors = (
        sentiment_logit_errors
        @ hidden_to_sentiment_weights.T
    )

    return (
        sentiment_logit_errors,
        hidden_to_sentiment_weight_gradients,
        sentiment_output_bias_gradients,
        final_hidden_state_errors,
    )


def backpropagate_through_sentence(
    all_hidden_states,
    actual_sequence_lengths,
    final_hidden_state_errors,
    hidden_to_hidden_weights,
):
    """
    Performs Backpropagation Through Time (BPTT) for a Many-to-One sequence.

    Key Many-to-One BPTT Mechanism:
        Unlike Many-to-Many models (which receive an output error at every timestep),
        a Many-to-One model receives a direct error ONLY at the true final timestep t = L-1:
            delta_{h, t}^{direct} = delta_{h, L}^{final}  if t == L - 1
            delta_{h, t}^{direct} = 0.0                   if t != L - 1

        For padding timesteps (t >= L):
            Since h_t was frozen (h_t = h_{t-1}), the gradient through tanh is zero:
            delta_{z, t} = 0.0

        For real word timesteps (t < L):
            delta_{z, t} = (delta_{h, t}^{direct} + delta_{h, t}^{future}) * (1 - h_{t+1}^2)
            delta_{h, t-1}^{future} = delta_{z, t} @ (W_{hh})^T

    Parameters:
        all_hidden_states (np.ndarray): History [h_0, ..., h_T]. Shape: (B, T+1, H)
        actual_sequence_lengths (np.ndarray): True lengths L. Shape: (B,)
        final_hidden_state_errors (np.ndarray): delta_{h, L}^{final}. Shape: (B, H)
        hidden_to_hidden_weights (np.ndarray): W_{hh}. Shape: (H, H)

    Returns:
        tuple (total_hidden_state_errors, hidden_pre_activation_errors):
            - total_hidden_state_errors (np.ndarray): Shape: (B, T, H)
            - hidden_pre_activation_errors (np.ndarray): delta_{z, t}. Shape: (B, T, H)
    """
    batch_size, states_including_initial_state, hidden_state_size = (
        all_hidden_states.shape
    )

    maximum_sequence_length = (
        states_including_initial_state - 1
    )

    total_hidden_state_errors = np.zeros(
        (batch_size, maximum_sequence_length, hidden_state_size)
    )

    hidden_pre_activation_errors = np.zeros(
        (batch_size, maximum_sequence_length, hidden_state_size)
    )

    future_hidden_state_errors = np.zeros(
        (batch_size, hidden_state_size)
    )

    # Step backwards in time: T-1, T-2, ..., 0
    for timestep_index in range(
        maximum_sequence_length - 1,
        -1,
        -1,
    ):
        current_hidden_state = all_hidden_states[
            :,
            timestep_index + 1,
            :,
        ]

        # 1. Direct output error is injected strictly at the true sentence boundary (t == L - 1)
        is_final_meaningful_timestep = (
            timestep_index
            == actual_sequence_lengths - 1
        )[:, np.newaxis]

        direct_final_state_error = np.where(
            is_final_meaningful_timestep,
            final_hidden_state_errors,
            0.0,
        )

        # 2. Total error on hidden state: direct error + recurrence from future timesteps
        total_hidden_state_error = (
            direct_final_state_error
            + future_hidden_state_errors
        )

        # 3. Tanh derivative: 1 - h^2
        tanh_derivative = (
            1.0 - current_hidden_state ** 2
        )

        # 4. Zero out gradients on padded timesteps (t >= L)
        is_real_word_timestep = (
            timestep_index < actual_sequence_lengths
        )[:, np.newaxis]

        hidden_pre_activation_error = (
            total_hidden_state_error
            * tanh_derivative
            * is_real_word_timestep
        )

        total_hidden_state_errors[
            :,
            timestep_index,
            :,
        ] = total_hidden_state_error

        hidden_pre_activation_errors[
            :,
            timestep_index,
            :,
        ] = hidden_pre_activation_error

        # 5. Compute recurrent error to pass to earlier timestep: delta_{z, t} @ W_{hh}^T
        future_hidden_state_errors = (
            hidden_pre_activation_error
            @ hidden_to_hidden_weights.T
        )

    return (
        total_hidden_state_errors,
        hidden_pre_activation_errors,
    )


def recurrent_and_embedding_gradients(
    token_id_batch,
    word_embedding_batch,
    all_hidden_states,
    hidden_pre_activation_errors,
    embedding_to_hidden_weights,
    embedding_matrix,
    padding_token_index,
):
    """
    Computes parameter gradients for recurrent matrices and the embedding table E.

    Mathematical Formulation:
        1. Embedding-to-Hidden Weights:
           dL / dW_{xh} = sum_{b, t} (e_{b, t})^T @ delta_{z, b, t} \in R^{D x H}
        2. Recurrent Transition Weights:
           dL / dW_{hh} = sum_{b, t} (h_{b, t-1})^T @ delta_{z, b, t} \in R^{H x H}
        3. Hidden Bias:
           dL / db_h = sum_{b, t} delta_{z, b, t} \in R^{1 x H}
        4. Word Embedding Vectors:
           delta_{e, b, t} = delta_{z, b, t} @ (W_{xh})^T \in R^{1 x D}
        5. Embedding Matrix Row Accumulation (Scatter-Add):
           For each token index w != <PAD>:
               dL / dE[w] = sum_{(b, t): token_{b, t} == w} delta_{e, b, t} \in R^{1 x D}

    Parameters:
        token_id_batch (np.ndarray): Padded token indices. Shape: (B, T)
        word_embedding_batch (np.ndarray): Embeddings e_{b, t}. Shape: (B, T, D)
        all_hidden_states (np.ndarray): History [h_0, ..., h_T]. Shape: (B, T+1, H)
        hidden_pre_activation_errors (np.ndarray): delta_{z, t}. Shape: (B, T, H)
        embedding_to_hidden_weights (np.ndarray): W_{xh}. Shape: (D, H)
        embedding_matrix (np.ndarray): Current table E. Shape: (V, D)
        padding_token_index (int): Index of <PAD> (0).

    Returns:
        tuple (embedding_to_hidden_weight_gradients, hidden_to_hidden_weight_gradients,
               hidden_bias_gradients, word_embedding_errors, embedding_matrix_gradients)
    """
    batch_size, maximum_sequence_length, embedding_dimension = (
        word_embedding_batch.shape
    )

    hidden_state_size = all_hidden_states.shape[2]

    embedding_to_hidden_weight_gradients = np.zeros(
        (embedding_dimension, hidden_state_size)
    )

    hidden_to_hidden_weight_gradients = np.zeros(
        (hidden_state_size, hidden_state_size)
    )

    hidden_bias_gradients = np.zeros(
        (1, hidden_state_size)
    )

    word_embedding_errors = np.zeros_like(
        word_embedding_batch
    )

    # Accumulate recurrent parameter gradients across all timesteps
    for timestep_index in range(
        maximum_sequence_length
    ):
        current_word_embeddings = word_embedding_batch[
            :,
            timestep_index,
            :,
        ]

        previous_hidden_states = all_hidden_states[
            :,
            timestep_index,
            :,
        ]

        current_hidden_pre_activation_errors = (
            hidden_pre_activation_errors[
                :,
                timestep_index,
                :,
            ]
        )

        embedding_to_hidden_weight_gradients += (
            current_word_embeddings.T
            @ current_hidden_pre_activation_errors
        )

        hidden_to_hidden_weight_gradients += (
            previous_hidden_states.T
            @ current_hidden_pre_activation_errors
        )

        hidden_bias_gradients += np.sum(
            current_hidden_pre_activation_errors,
            axis=0,
            keepdims=True,
        )

        # Backpropagate into embedding vectors: delta_e = delta_z @ W_{xh}^T
        word_embedding_errors[
            :,
            timestep_index,
            :,
        ] = (
            current_hidden_pre_activation_errors
            @ embedding_to_hidden_weights.T
        )

    # Scatter-add gradients into the master embedding matrix E
    embedding_matrix_gradients = np.zeros_like(
        embedding_matrix
    )

    for batch_index in range(batch_size):
        for timestep_index in range(
            maximum_sequence_length
        ):
            current_token_index = token_id_batch[
                batch_index,
                timestep_index,
            ]

            # Only update embedding weights for real words, skipping <PAD>
            if current_token_index != padding_token_index:
                embedding_matrix_gradients[
                    current_token_index
                ] += word_embedding_errors[
                    batch_index,
                    timestep_index,
                ]

    return (
        embedding_to_hidden_weight_gradients,
        hidden_to_hidden_weight_gradients,
        hidden_bias_gradients,
        word_embedding_errors,
        embedding_matrix_gradients,
    )


def sentiment_rnn_backward(
    token_id_batch,
    actual_sequence_lengths,
    sentiment_labels,
    forward_pass_results,
    model_parameters,
    padding_token_index,
):
    """
    Assembles the complete backward pass: combines output head, BPTT, and embedding gradients.

    Returns:
        dict: Gradients for all 6 learnable parameters:
            - 'embedding_matrix': (V, D)
            - 'embedding_to_hidden_weights': (D, H)
            - 'hidden_to_hidden_weights': (H, H)
            - 'hidden_bias': (1, H)
            - 'hidden_to_sentiment_weights': (H, 1)
            - 'sentiment_output_bias': (1, 1)
    """
    # 1. Output classification head gradients
    (
        _,
        hidden_to_sentiment_weight_gradients,
        sentiment_output_bias_gradients,
        final_hidden_state_errors,
    ) = sentiment_output_backward(
        forward_pass_results["final_hidden_states"],
        forward_pass_results[
            "positive_sentiment_probabilities"
        ],
        sentiment_labels,
        model_parameters[
            "hidden_to_sentiment_weights"
        ],
    )

    # 2. Backpropagation through time across the unrolled sentence
    (
        _,
        hidden_pre_activation_errors,
    ) = backpropagate_through_sentence(
        forward_pass_results["all_hidden_states"],
        actual_sequence_lengths,
        final_hidden_state_errors,
        model_parameters[
            "hidden_to_hidden_weights"
        ],
    )

    # 3. Recurrent layer and word embedding table gradients
    (
        embedding_to_hidden_weight_gradients,
        hidden_to_hidden_weight_gradients,
        hidden_bias_gradients,
        _,
        embedding_matrix_gradients,
    ) = recurrent_and_embedding_gradients(
        token_id_batch,
        forward_pass_results["word_embedding_batch"],
        forward_pass_results["all_hidden_states"],
        hidden_pre_activation_errors,
        model_parameters[
            "embedding_to_hidden_weights"
        ],
        model_parameters["embedding_matrix"],
        padding_token_index,
    )

    return {
        "embedding_matrix": embedding_matrix_gradients,
        "embedding_to_hidden_weights": (
            embedding_to_hidden_weight_gradients
        ),
        "hidden_to_hidden_weights": (
            hidden_to_hidden_weight_gradients
        ),
        "hidden_bias": hidden_bias_gradients,
        "hidden_to_sentiment_weights": (
            hidden_to_sentiment_weight_gradients
        ),
        "sentiment_output_bias": (
            sentiment_output_bias_gradients
        ),
    }


# =============================================================================
# PART 6: OPTIMIZATION & TRAINING LOOP
# =============================================================================

def clip_gradients_by_global_norm(
    parameter_gradients,
    maximum_gradient_norm,
):
    """
    Clips parameter gradients by global L2 norm to prevent gradient explosion.

    Mathematical Formulation:
        ||g|| = sqrt( sum_{p} sum_{i, j} (g_{p, i, j})^2 )
        scale = min(1.0, max_norm / (||g|| + 1e-12))
        g_p^{clipped} = g_p * scale

    Parameters:
        parameter_gradients (dict): Mapping from parameter name to gradient array.
        maximum_gradient_norm (float): Maximum allowed L2 norm threshold (e.g., 5.0).

    Returns:
        tuple (clipped_parameter_gradients, global_gradient_norm)
    """
    total_squared_gradient_norm = sum(
        np.sum(parameter_gradient ** 2)
        for parameter_gradient
        in parameter_gradients.values()
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
        parameter_name: (
            parameter_gradient * clipping_scale
        )
        for parameter_name, parameter_gradient
        in parameter_gradients.items()
    }

    return (
        clipped_parameter_gradients,
        global_gradient_norm,
    )


def gradient_descent_update(
    model_parameters,
    parameter_gradients,
    learning_rate,
):
    """
    Updates model parameters using standard Stochastic Gradient Descent (SGD).

    Mathematical Formulation:
        theta_{new} = theta_{old} - eta * dL / d(theta)

    Parameters:
        model_parameters (dict): Weights and biases
        parameter_gradients (dict): Clipped gradients
        learning_rate (float): Step size eta (e.g., 0.1)

    Returns:
        updated_model_parameters (dict): Parameter dictionary after update
    """
    updated_model_parameters = {
        parameter_name: (
            parameter_value
            - learning_rate
            * parameter_gradients[parameter_name]
        )
        for parameter_name, parameter_value
        in model_parameters.items()
    }

    return updated_model_parameters


def train_sentiment_rnn(
    training_token_id_batch,
    training_sequence_lengths,
    training_sentiment_labels,
    initial_model_parameters,
    learning_rate,
    number_of_epochs,
):
    """
    Trains the Many-to-One sentiment RNN over the educational batch of reviews.

    Training Loop (per epoch):
        1. Forward Pass: lookup embeddings, unroll RNN, compute sigmoid probabilities
        2. Evaluate Loss: binary cross-entropy
        3. Backward Pass: BPTT + embedding gradients
        4. Gradient Clipping: threshold at 5.0
        5. Gradient Descent Update: step size eta

    Parameters:
        training_token_id_batch (np.ndarray): Padded batch matrix (B, T)
        training_sequence_lengths (np.ndarray): True sentence lengths (B,)
        training_sentiment_labels (np.ndarray): Binary labels (B, 1)
        initial_model_parameters (dict): Model parameter dictionary
        learning_rate (float): Learning rate eta (e.g., 0.1)
        number_of_epochs (int): Number of training iterations (e.g., 1500)

    Returns:
        tuple (current_model_parameters, loss_history):
            - current_model_parameters (dict): Trained model weights
            - loss_history (list): Loss value per epoch
    """
    current_model_parameters = {
        parameter_name: parameter_value.copy()
        for parameter_name, parameter_value
        in initial_model_parameters.items()
    }

    batch_size = training_token_id_batch.shape[0]

    hidden_state_size = (
        current_model_parameters[
            "hidden_to_hidden_weights"
        ].shape[0]
    )

    initial_hidden_state = np.zeros(
        (batch_size, hidden_state_size)
    )

    padding_token_index = 0

    loss_history = []

    for epoch_index in range(number_of_epochs):
        # 1. Forward Pass
        forward_pass_results = sentiment_rnn_forward(
            training_token_id_batch,
            training_sequence_lengths,
            initial_hidden_state,
            current_model_parameters,
        )

        # 2. Binary Cross-Entropy Loss
        current_loss = binary_cross_entropy(
            forward_pass_results[
                "positive_sentiment_probabilities"
            ],
            training_sentiment_labels,
        )

        # 3. Backward Pass (BPTT)
        parameter_gradients = sentiment_rnn_backward(
            training_token_id_batch,
            training_sequence_lengths,
            training_sentiment_labels,
            forward_pass_results,
            current_model_parameters,
            padding_token_index,
        )

        # 4. Global Norm Gradient Clipping
        clipped_parameter_gradients, _ = (
            clip_gradients_by_global_norm(
                parameter_gradients,
                maximum_gradient_norm=5.0,
            )
        )

        # 5. Parameter Update
        current_model_parameters = gradient_descent_update(
            current_model_parameters,
            clipped_parameter_gradients,
            learning_rate,
        )

        loss_history.append(current_loss)

        if epoch_index % 100 == 0:
            print(
                f"Epoch {epoch_index:4d} | "
                f"Loss: {current_loss:.6f}"
            )

    return current_model_parameters, loss_history


def predict_sentiment_labels(
    token_id_batch,
    actual_sequence_lengths,
    trained_model_parameters,
):
    """
    Evaluates predictions and thresholds probabilities at 0.5 for binary classification.

    Classification Rule:
        y_pred = 1.0 (Positive)  if y_hat >= 0.5
        y_pred = 0.0 (Negative)  if y_hat < 0.5

    Parameters:
        token_id_batch (np.ndarray): Padded token indices. Shape: (B, T)
        actual_sequence_lengths (np.ndarray): True sentence lengths. Shape: (B,)
        trained_model_parameters (dict): Trained model weights

    Returns:
        tuple (positive_sentiment_probabilities, predicted_sentiment_labels):
            - positive_sentiment_probabilities (np.ndarray): Continuous probabilities in (0, 1)
            - predicted_sentiment_labels (np.ndarray): Binary predictions in {0.0, 1.0}
    """
    batch_size = token_id_batch.shape[0]

    hidden_state_size = (
        trained_model_parameters[
            "hidden_to_hidden_weights"
        ].shape[0]
    )

    initial_hidden_state = np.zeros(
        (batch_size, hidden_state_size)
    )

    forward_pass_results = sentiment_rnn_forward(
        token_id_batch,
        actual_sequence_lengths,
        initial_hidden_state,
        trained_model_parameters,
    )

    positive_sentiment_probabilities = (
        forward_pass_results[
            "positive_sentiment_probabilities"
        ]
    )

    predicted_sentiment_labels = (
        positive_sentiment_probabilities >= 0.5
    ).astype(np.float64)

    return (
        positive_sentiment_probabilities,
        predicted_sentiment_labels,
    )


# =============================================================================
# PART 7: MATHEMATICAL VERIFICATION (FINITE-DIFFERENCE GRADIENT CHECKING)
# =============================================================================

def calculate_sentiment_batch_loss(
    token_id_batch,
    actual_sequence_lengths,
    sentiment_labels,
    model_parameters,
):
    """
    Helper function evaluating binary cross-entropy loss for perturbed parameter states.
    """
    batch_size = token_id_batch.shape[0]

    hidden_state_size = (
        model_parameters[
            "hidden_to_hidden_weights"
        ].shape[0]
    )

    initial_hidden_state = np.zeros(
        (batch_size, hidden_state_size)
    )

    forward_pass_results = sentiment_rnn_forward(
        token_id_batch,
        actual_sequence_lengths,
        initial_hidden_state,
        model_parameters,
    )

    return binary_cross_entropy(
        forward_pass_results[
            "positive_sentiment_probabilities"
        ],
        sentiment_labels,
    )


def numerical_gradient_for_parameter_element(
    token_id_batch,
    actual_sequence_lengths,
    sentiment_labels,
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

    Verification Purpose:
        Validates the analytical BPTT calculus through the embedding lookup and recurrent loop.
        Because two-sided finite difference has approximation error O(eps^2), matching
        within 1e-4 provides mathematical proof of backpropagation correctness.

    Parameters:
        token_id_batch (np.ndarray): Padded input batch
        actual_sequence_lengths (np.ndarray): True sentence lengths
        sentiment_labels (np.ndarray): True binary labels
        model_parameters (dict): Current parameter dictionary
        parameter_name (str): Key of parameter to test (e.g., 'embedding_matrix')
        row_index (int): Row index in parameter matrix
        column_index (int): Column index in parameter matrix
        epsilon (float): Finite difference perturbation (default: 1e-5)

    Returns:
        numerical_gradient (float): Empirical derivative
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

    positive_loss = calculate_sentiment_batch_loss(
        token_id_batch,
        actual_sequence_lengths,
        sentiment_labels,
        parameters_with_positive_change,
    )

    negative_loss = calculate_sentiment_batch_loss(
        token_id_batch,
        actual_sequence_lengths,
        sentiment_labels,
        parameters_with_negative_change,
    )

    numerical_gradient = (
        positive_loss - negative_loss
    ) / (2.0 * epsilon)

    return numerical_gradient


# =============================================================================
# PART 8: EXECUTABLE EDUCATIONAL WALKTHROUGH & VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("==================================================================")
    print(" 08 — Many-to-One Sentiment Analysis RNN From Scratch Walkthrough ")
    print("==================================================================")

    # -------------------------------------------------------------------------
    # Step 1: Preprocess Walkthrough Review "Acting was not good!"
    # -------------------------------------------------------------------------
    review_text = "Acting was not good!"
    word_tokens = normalize_and_tokenize(review_text)

    word_to_index = create_teaching_vocabulary()
    token_ids = convert_tokens_to_token_ids(
        word_tokens,
        word_to_index,
    )

    padding_token_index = word_to_index["<PAD>"]
    padded_token_ids, actual_sequence_length = (
        pad_or_truncate_token_ids(
            token_ids,
            maximum_sequence_length=6,
            padding_token_index=padding_token_index,
        )
    )

    # -------------------------------------------------------------------------
    # Step 2: Continuous Word Embedding Lookup
    # -------------------------------------------------------------------------
    embedding_matrix = create_teaching_embedding_matrix()
    token_id_batch = np.array(
        [padded_token_ids],
        dtype=np.int64,
    )
    word_embedding_batch = lookup_word_embeddings(
        token_id_batch,
        embedding_matrix,
    )

    # -------------------------------------------------------------------------
    # Step 3: Single Recurrent Cell Step (First Token: "acting")
    # -------------------------------------------------------------------------
    (
        embedding_to_hidden_weights,
        hidden_to_hidden_weights,
        hidden_bias,
    ) = create_teaching_recurrent_parameters()

    initial_hidden_state = np.zeros((1, 3))
    current_word_embedding = word_embedding_batch[
        :,
        0,
        :,
    ]

    current_hidden_state, hidden_pre_activation = (
        sentiment_rnn_cell_forward(
            current_word_embedding,
            initial_hidden_state,
            embedding_to_hidden_weights,
            hidden_to_hidden_weights,
            hidden_bias,
        )
    )

    # -------------------------------------------------------------------------
    # Step 4: Many-to-One Forward Pass with Padding Freezing
    # -------------------------------------------------------------------------
    actual_sequence_lengths = np.array(
        [actual_sequence_length],
        dtype=np.int64,
    )

    all_hidden_states, final_hidden_states = (
        many_to_one_rnn_forward(
            word_embedding_batch,
            actual_sequence_lengths,
            initial_hidden_state,
            embedding_to_hidden_weights,
            hidden_to_hidden_weights,
            hidden_bias,
        )
    )

    # -------------------------------------------------------------------------
    # Step 5: Sentiment Classification Head & Loss
    # -------------------------------------------------------------------------
    hidden_to_sentiment_weights = np.array([
        [0.5],
        [-0.4],
        [0.3],
    ])
    sentiment_output_bias = np.array([[0.1]])

    (
        sentiment_logits,
        positive_sentiment_probabilities,
    ) = sentiment_output_forward(
        final_hidden_states,
        hidden_to_sentiment_weights,
        sentiment_output_bias,
    )

    # Ground truth label for "acting was not good" is NEGATIVE (0.0)
    sentiment_labels = np.array([[0.0]])
    binary_cross_entropy_loss = binary_cross_entropy(
        positive_sentiment_probabilities,
        sentiment_labels,
    )

    # -------------------------------------------------------------------------
    # Step 6: Many-to-One BPTT & Parameter Gradients
    # -------------------------------------------------------------------------
    (
        sentiment_logit_errors,
        hidden_to_sentiment_weight_gradients,
        sentiment_output_bias_gradients,
        final_hidden_state_errors,
    ) = sentiment_output_backward(
        final_hidden_states,
        positive_sentiment_probabilities,
        sentiment_labels,
        hidden_to_sentiment_weights,
    )

    (
        total_hidden_state_errors,
        hidden_pre_activation_errors,
    ) = backpropagate_through_sentence(
        all_hidden_states,
        actual_sequence_lengths,
        final_hidden_state_errors,
        hidden_to_hidden_weights,
    )

    (
        embedding_to_hidden_weight_gradients,
        hidden_to_hidden_weight_gradients,
        hidden_bias_gradients,
        word_embedding_errors,
        embedding_matrix_gradients,
    ) = recurrent_and_embedding_gradients(
        token_id_batch,
        word_embedding_batch,
        all_hidden_states,
        hidden_pre_activation_errors,
        embedding_to_hidden_weights,
        embedding_matrix,
        padding_token_index,
    )

    model_parameters = {
        "embedding_matrix": embedding_matrix,
        "embedding_to_hidden_weights": (
            embedding_to_hidden_weights
        ),
        "hidden_to_hidden_weights": (
            hidden_to_hidden_weights
        ),
        "hidden_bias": hidden_bias,
        "hidden_to_sentiment_weights": (
            hidden_to_sentiment_weights
        ),
        "sentiment_output_bias": sentiment_output_bias,
    }

    complete_forward_results = sentiment_rnn_forward(
        token_id_batch,
        actual_sequence_lengths,
        initial_hidden_state,
        model_parameters,
    )

    complete_parameter_gradients = (
        sentiment_rnn_backward(
            token_id_batch,
            actual_sequence_lengths,
            sentiment_labels,
            complete_forward_results,
            model_parameters,
            padding_token_index,
        )
    )

    # -------------------------------------------------------------------------
    # Step 7: Gradient Clipping & One Optimization Step
    # -------------------------------------------------------------------------
    (
        clipped_parameter_gradients,
        global_gradient_norm,
    ) = clip_gradients_by_global_norm(
        complete_parameter_gradients,
        maximum_gradient_norm=5.0,
    )

    updated_model_parameters = gradient_descent_update(
        model_parameters,
        clipped_parameter_gradients,
        learning_rate=0.01,
    )

    updated_forward_results = sentiment_rnn_forward(
        token_id_batch,
        actual_sequence_lengths,
        initial_hidden_state,
        updated_model_parameters,
    )

    updated_binary_cross_entropy_loss = (
        binary_cross_entropy(
            updated_forward_results[
                "positive_sentiment_probabilities"
            ],
            sentiment_labels,
        )
    )

    # -------------------------------------------------------------------------
    # Step 8: Train Model on 8-Review Dataset
    # -------------------------------------------------------------------------
    training_review_texts = [
        "acting was good",          # Positive (1)
        "movie was great",          # Positive (1)
        "the film was great",       # Positive (1)
        "movie was not terrible",   # Positive (1)
        "acting was terrible",      # Negative (0)
        "movie was bad",            # Negative (0)
        "the film was boring",      # Negative (0)
        "acting was not good",      # Negative (0)
    ]

    training_sentiment_labels = np.array([
        [1.0],
        [1.0],
        [1.0],
        [1.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
    ])

    training_token_id_batch, training_sequence_lengths = (
        prepare_sentiment_batch(
            training_review_texts,
            word_to_index,
        )
    )

    print("\n--- Training Sentiment RNN (1500 Epochs) ---")
    (
        trained_sentiment_model_parameters,
        sentiment_loss_history,
    ) = train_sentiment_rnn(
        training_token_id_batch,
        training_sequence_lengths,
        training_sentiment_labels,
        model_parameters,
        learning_rate=0.1,
        number_of_epochs=1500,
    )

    # -------------------------------------------------------------------------
    # Step 9: Evaluate Batch Predictions & Accuracy
    # -------------------------------------------------------------------------
    (
        training_positive_probabilities,
        predicted_training_labels,
    ) = predict_sentiment_labels(
        training_token_id_batch,
        training_sequence_lengths,
        trained_sentiment_model_parameters,
    )

    training_accuracy = np.mean(
        predicted_training_labels
        == training_sentiment_labels
    )

    print("\n--- Model Predictions on Training Batch ---")
    for review_text, true_label, probability, predicted_label in zip(
        training_review_texts,
        training_sentiment_labels,
        training_positive_probabilities,
        predicted_training_labels,
    ):
        print(
            f"{review_text!r:25s} | "
            f"true={int(true_label[0])} | "
            f"prob={probability[0]:.4f} | "
            f"pred={int(predicted_label[0])}"
        )

    # -------------------------------------------------------------------------
    # Step 10: Finite-Difference Gradient Check on Embedding Table
    # -------------------------------------------------------------------------
    analytical_embedding_gradient = (
        complete_parameter_gradients[
            "embedding_matrix"
        ][word_to_index["good"], 0]
    )

    numerical_embedding_gradient = (
        numerical_gradient_for_parameter_element(
            token_id_batch,
            actual_sequence_lengths,
            sentiment_labels,
            model_parameters,
            parameter_name="embedding_matrix",
            row_index=word_to_index["good"],
            column_index=0,
        )
    )

    # -------------------------------------------------------------------------
    # Step 11: Print Verification Summary
    # -------------------------------------------------------------------------
    print("\n--- Single Sentence Walkthrough Verification ---")
    print("Word tokens:                ", word_tokens)
    print("Token IDs:                  ", token_ids)
    print("Padded token IDs:           ", padded_token_ids)
    print("Actual sequence length:     ", actual_sequence_length)
    print("Word embedding batch shape: ", word_embedding_batch.shape)
    print("Word embeddings:\n", word_embedding_batch)
    print("Hidden pre-activation z_0:  ", hidden_pre_activation)
    print("Current hidden state h_0:   ", current_hidden_state)
    print("All hidden states:\n", all_hidden_states)
    print("Final meaningful state h_L: ", final_hidden_states)
    print("Sentiment logit o:          ", sentiment_logits)
    print("Positive probability y_hat: ", positive_sentiment_probabilities)
    print("Binary cross-entropy loss:  ", binary_cross_entropy_loss)

    print("\n--- Backward Pass & Gradient Verification ---")
    print("Sentiment logit errors delta_o:      ", sentiment_logit_errors)
    print("Hidden-to-sentiment weight grads:    \n", hidden_to_sentiment_weight_gradients)
    print("Sentiment output bias grads:         ", sentiment_output_bias_gradients)
    print("Final hidden-state errors:           ", final_hidden_state_errors)
    print("Hidden pre-activation errors:\n", hidden_pre_activation_errors)
    print("Embedding-to-hidden weight grads:    \n", embedding_to_hidden_weight_gradients)
    print("Hidden-to-hidden weight grads:       \n", hidden_to_hidden_weight_gradients)
    print("Hidden bias gradients:               ", hidden_bias_gradients)
    print("Embedding gradient for 'good':       ", embedding_matrix_gradients[word_to_index["good"]])
    print("Global gradient norm ||g||:          ", global_gradient_norm)
    print("Loss before update:                  ", binary_cross_entropy_loss)
    print("Loss after one update:               ", updated_binary_cross_entropy_loss)

    print("\n--- Batch Training Summary ---")
    print("Training token batch shape: ", training_token_id_batch.shape)
    print("Training sequence lengths:  ", training_sequence_lengths)
    print("Initial batch loss:         ", sentiment_loss_history[0])
    print("Final batch loss:           ", sentiment_loss_history[-1])
    print("Training accuracy:          ", training_accuracy)

    print("\n--- Embedding Gradient Verification (Analytical vs Numerical) ---")
    print("Analytical embedding gradient dE[good, 0]:", analytical_embedding_gradient)
    print("Numerical embedding gradient  dE[good, 0]:", numerical_embedding_gradient)
    embedding_gradients_match = np.isclose(
        analytical_embedding_gradient,
        numerical_embedding_gradient,
        rtol=1e-4,
        atol=1e-6,
    )
    print("Gradients match within tolerance:", embedding_gradients_match)
    print("==================================================================\n")