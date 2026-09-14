import numpy as np

# The tokenizer is vendored into this project (see bpe/VENDORED.md), so it
# imports like any other local package -- no sys.path juggling, and none of
# its FastAPI/SSE layer comes along.
from bpe.tokenizer import Tokenizer
from step_02_attention import AttentionHead
from step_03_multihead import MultiHeadAttention
from step_04_feedforward import FeedForward
from step_05_transformer_block import TransformerBlock
from step_06_language_model import MiniLLM
from step_07_loss import cross_entropy
from step_08_gradients import loss_and_gradient
from positions import positional_encoding

def main() -> None:
    # 1. Configuration.
    sequence_length = 16
    d_model = 64
    rng = np.random.default_rng(42)

    # 2. Use your tokenizer in its default byte-level mode.
    tokenizer = Tokenizer()
    text = (
        "Transformers learn from sequences. "
        "Each position predicts the next token."
    ) 

    encoded = tokenizer.encode(text)
    token_ids = np.asarray(
        encoded.token_ids, dtype=np.int64
    )       

    assert (
        tokenizer.decode(token_ids.tolist())
        == encoded.normalized_text
    )

    # 3. Confirm vocabulary IDs can index a dense table.
    vocab_size = tokenizer.vocab_size
    actual_ids = set(tokenizer.vocabulary.id_to_token)
    assert actual_ids == set(range(vocab_size))  

    # We need T + 1 tokens to create T prediction targets.
    if len(token_ids) < sequence_length + 1:
        raise ValueError("Provide more text for this sequence length")

    # 4. Build one training example.
    input_ids = token_ids[:sequence_length][None, :]
    target_ids = token_ids[1:sequence_length + 1][None, :]

    # Both have shape (B=1, T=16).
    assert np.array_equal(
        input_ids[:, 1:],
        target_ids[:, :-1],
    )

    # 5. Initialize token embeddings: shape (V, d_model).
    embedding_table = rng.normal(
        loc=0.0,
        scale=0.02,
        size=(vocab_size, d_model),
    ).astype(np.float32)

    # 6. Look up embeddings: shape (B, T, d_model).
    token_vectors = embedding_table[input_ids]

    # 7. Create positions: shape (T, d_model).
    position_vectors = positional_encoding(
        sequence_length, d_model
    )

    # 8. Add positions to every sequence in the batch.
    x = token_vectors + position_vectors[None, :, :]

    assert x.shape == (1, sequence_length, d_model)
    assert np.isfinite(x).all()   


    # -----------------  step_02_attention
    # Apply one attention head to the embeddings.
    head = AttentionHead(d_model=d_model, head_dim=16)
    attention_output, attention_weights = head.forward(x)

    # Every query distributes a total weight of 1.
    assert np.allclose(
        attention_weights.sum(axis=-1),
        1.0,
    )

    # No query assigns weight to a future position.
    assert np.all(
        np.triu(attention_weights, k=1) == 0
    )

    # Changing the final token must not affect earlier outputs.
    changed_x = x.copy()
    changed_x[:, -1, :] += 10.0
    changed_output, _ = head.forward(changed_x)

    assert np.allclose(
        attention_output[:, :-1, :],
        changed_output[:, :-1, :],
    )


    # -----------------  step_03_multihead
    # Apply four heads to the original embeddings.
    mha = MultiHeadAttention(
        d_model=d_model,
        num_heads=4,
    )
    multihead_output, multihead_weights = mha.forward(x)

    assert multihead_output.shape == x.shape
    assert multihead_weights.shape == (
        x.shape[0], 4, sequence_length, sequence_length
    )

    # Every head must preserve the attention constraints.
    assert np.allclose(
        multihead_weights.sum(axis=-1),
        1.0,
    )
    assert np.all(
        np.triu(multihead_weights, k=1) == 0
    )

    # The output projection must preserve causality too.
    changed_x = x.copy()
    changed_x[:, -1, :] += 10.0
    changed_output, _ = mha.forward(changed_x)

    assert np.allclose(
        multihead_output[:, :-1, :],
        changed_output[:, :-1, :],
    )


    # ----------------- step_04_feedforward
    ffn = FeedForward(
        d_model=d_model,
        d_ff=4 * d_model,
    )

    ffn_output = ffn.forward(multihead_output)

    assert ffn_output.shape == multihead_output.shape
    assert np.isfinite(ffn_output).all()

    # Change one position at the FFN's input.
    changed_input = multihead_output.copy()
    changed_input[:, 5, :] += 10.0
    changed_ffn_output = ffn.forward(changed_input)

    # Every other position must remain unchanged.
    other_positions = np.arange(sequence_length) != 5

    assert np.allclose(
        ffn_output[:, other_positions, :],
        changed_ffn_output[:, other_positions, :],
    )


    # ----------------- step_05_transformer_block
    block = TransformerBlock(
        d_model=d_model,
        num_heads=4,
        d_ff=4 * d_model,
    )

    # Start from the original token + position embeddings.
    block_output, block_weights = block.forward(x)

    assert block_output.shape == x.shape
    assert np.isfinite(block_output).all()

    # At initialization gamma=1 and beta=0.
    normalized = block.norm1.forward(x)
    assert np.allclose(
        normalized.mean(axis=-1),
        0.0,
        atol=1e-5,
    )

    # A constant feature vector must also normalize safely.
    constant_output = block.norm1.forward(np.ones_like(x))
    assert np.allclose(constant_output, 0.0)

    # Change the final position using a nonuniform perturbation.
    changed_x = x.copy()
    changed_x[:, -1, :] += np.linspace(
        -1.0, 1.0, d_model, dtype=np.float32
    )

    changed_block_output, _ = block.forward(changed_x)

    # Earlier positions must remain unaffected.
    assert np.allclose(
        block_output[:, :-1, :],
        changed_block_output[:, :-1, :],
        atol=1e-6,
    )


    # ----------------- step_06_language_model
    model = MiniLLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=4,
        num_layers=2,
        max_seq_len=64,
    )

    # The complete model takes integer IDs.
    logits, layer_attention = model.forward(input_ids)

    assert logits.shape == (
        input_ids.shape[0],
        sequence_length,
        vocab_size,
    )
    assert np.isfinite(logits).all()
    assert len(layer_attention) == 2

    # Changing the final input token must not change
    # predictions made at earlier positions.
    changed_ids = input_ids.copy()
    changed_ids[:, -1] = (
        changed_ids[:, -1] + 1
    ) % vocab_size

    changed_logits, _ = model.forward(changed_ids)

    assert np.allclose(
        logits[:, :-1, :],
        changed_logits[:, :-1, :],
        atol=1e-6,
    )


    # ----------------- step_07_loss
    loss = cross_entropy(logits, target_ids)

    # Equal scores give a uniform vocabulary distribution.
    uniform_logits = np.zeros_like(logits)
    uniform_loss = cross_entropy(
        uniform_logits, target_ids
    )

    assert np.isclose(
        uniform_loss,
        np.log(vocab_size),
    )

    # Diagnostic: increasing the correct scores must reduce loss.
    boosted_logits = logits.copy()

    batch_indices = np.arange(
        logits.shape[0]
    )[:, None]

    position_indices = np.arange(
        logits.shape[1]
    )[None, :]

    boosted_logits[
        batch_indices,
        position_indices,
        target_ids,
    ] += 4.0

    boosted_loss = cross_entropy(
        boosted_logits, target_ids
    )

    assert boosted_loss < loss

    # Perplexity uses the same natural logarithm as the loss.
    perplexity = np.exp(loss)


    # ----------------- step_08_gradients
    scores64 = logits.astype(np.float64)

    checked_loss, gradient = loss_and_gradient(
        scores64, target_ids
    )

    assert np.isclose(checked_loss, loss)
    assert gradient.shape == logits.shape
    assert np.isfinite(gradient).all()

    # Within each position, the vocabulary gradients sum to zero:
    # sum(p - one_hot_target) = 1 - 1.
    assert np.allclose(
        gradient.sum(axis=-1),
        0.0,
        atol=1e-12,
    )

    first_target = int(target_ids[0, 0])
    last_target = int(target_ids[0, -1])

    check_indices = [
        (0, 0, first_target),
        (0, 0, (first_target + 1) % vocab_size),
        (0, sequence_length - 1, last_target),
    ]

    epsilon = 1e-5



    print("Vocabulary size:", vocab_size)
    print("Input IDs:", input_ids)
    print("Target IDs:", target_ids)
    print("Embedding table:", embedding_table.shape)
    print("Token vectors:", token_vectors.shape)
    print("Position vectors:", position_vectors.shape)
    print("Transformer input:", x.shape)
    print("First position, first 8 values:", x[0, 0, :8]) 


    print("Attention output:", attention_output.shape)
    print("Attention weights:", attention_weights.shape)
    print("First four positions:")
    print(np.round(attention_weights[0, :4, :4], 3))


    print("Multi-head output:", multihead_output.shape)
    print("Multi-head weights:", multihead_weights.shape)
    for i in range(4):
        print(f"Head {i + 1}, first four positions:")
        print(np.round(multihead_weights[0, i, :4, :4], 3))


    print("Feed-forward input:", multihead_output.shape)
    print("Expansion matrix:", ffn.W1.shape)
    print("Projection matrix:", ffn.W2.shape)
    print("Feed-forward output:", ffn_output.shape)
    print("Position independence check passed.")


    print("Transformer block input:", x.shape)
    print("Transformer block output:", block_output.shape)
    print("Block attention weights:", block_weights.shape)
    print("Layer normalization checks passed.")
    print("Block causality check passed.")  


    print("Model input IDs:", input_ids.shape)
    print("Vocabulary projection:", model.W_vocab.shape)
    print("Model logits:", logits.shape)
    for i, weights in enumerate(layer_attention):
        print(f"Layer {i + 1} attention:", weights.shape)
    print("Complete model causality check passed.")


    print(f"Model cross-entropy: {loss:.6f}")
    print(f"Uniform baseline: {uniform_loss:.6f}")
    print(f"Model perplexity: {perplexity:.3f}")
    print(
        "Loss after boosting correct scores: "
        f"{boosted_loss:.6f}"
    )
    print("Loss sanity checks passed.")   


    print("Loss gradient shape:", gradient.shape)
    for index in check_indices:
        plus = scores64.copy()
        minus = scores64.copy()

        plus[index] += epsilon
        minus[index] -= epsilon

        numerical = (
            cross_entropy(plus, target_ids)
            - cross_entropy(minus, target_ids)
        ) / (2.0 * epsilon)

        analytical = gradient[index]

        assert np.isclose(
            analytical,
            numerical,
            atol=1e-8,
            rtol=1e-5,
        ), f"Gradient check failed at {index}"

        print(
            f"{index}: "
            f"analytical={analytical:.8f}, "
            f"numerical={numerical:.8f}"
        )
    print("Gradient checks passed.")       

if __name__ == "__main__":
    main()