"""Step 9: extend a sequence one token at a time.

From 14-mini-llm:
    ../.venv/bin/python src/step_09_generate.py

This module works on token IDs only -- no tokenizer, no text. Turning a
string into IDs and back is Step 10's job, which is why the checks below
run against a freshly initialized model: the loop's correctness is a
structural property and does not depend on trained weights.
"""

import torch

from step_06_language_model import MiniLLM
from step_08_torch_model import TorchMiniLLM


def generate(
    model,
    token_ids,
    max_new_tokens,
    *,
    temperature: float = 0.0,
    top_k: int | None = None,
    generator=None,
):
    """Append `max_new_tokens` predicted IDs to a (B, T) prompt.

    temperature=0.0 selects the highest-scoring token, which makes the
    result reproducible. Any positive value samples instead; pass a
    `generator` on the same device as `token_ids` to make that repeatable
    too. top_k restricts sampling to the k best tokens and is applied
    after temperature, so the two compose in the usual order.

    Returns a new (B, T + max_new_tokens) tensor. The input is not mutated.
    """
    if token_ids.ndim != 2:
        raise ValueError("Expected token IDs with shape (batch, sequence)")

    if token_ids.dtype != torch.long:
        raise ValueError("Token IDs must be torch.long")

    if token_ids.shape[0] < 1 or token_ids.shape[1] < 1:
        raise ValueError("Prompt must contain at least one sequence and token")

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    if temperature < 0.0:
        raise ValueError("temperature must not be negative")

    if top_k is not None and not 1 <= top_k <= model.vocab_size:
        raise ValueError("top_k must fall between 1 and the vocabulary size")

    # Restore whatever mode the caller was in: generation is a read-only
    # operation and should not silently leave a training loop in eval().
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop BEFORE the forward pass, not after appending, so a
                # prompt that already exceeds the context window also works.
                # The model raises rather than truncating on its own.
                context = token_ids[:, -model.max_seq_len:]

                # (B, T, V) -- only the final position predicts a new token.
                # Every earlier position predicts one we already have.
                logits = model(context)
                next_logits = logits[:, -1, :]

                if temperature == 0.0:
                    # keepdim gives (B, 1) directly, ready to concatenate.
                    next_ids = next_logits.argmax(dim=-1, keepdim=True)
                else:
                    scaled = next_logits / temperature

                    if top_k is not None:
                        # topk returns values sorted high to low, so the last
                        # one is the cutoff every kept token must reach.
                        kept, _ = torch.topk(scaled, top_k, dim=-1)
                        cutoff = kept[:, -1:]
                        scaled = scaled.masked_fill(
                            scaled < cutoff, -torch.inf
                        )

                    probabilities = scaled.softmax(dim=-1)
                    next_ids = torch.multinomial(
                        probabilities, num_samples=1, generator=generator
                    )

                # multinomial and argmax both return torch.long, which the
                # model's dtype check requires on the next iteration.
                token_ids = torch.cat([token_ids, next_ids], dim=1)
    finally:
        model.train(was_training)

    return token_ids


def assert_rejects(call, description: str) -> None:
    """Confirm that an invalid call raises instead of guessing."""
    try:
        call()
    except ValueError:
        return

    raise AssertionError(f"Expected a ValueError: {description}")


def main() -> None:
    # 1. A fresh model. Untrained weights produce meaningless tokens, but
    #    every property checked below is about the loop, not the weights.
    torch.set_num_threads(1)
    vocab_size = 256
    max_seq_len = 64

    source = MiniLLM(
        vocab_size=vocab_size,
        d_model=64,
        num_heads=4,
        num_layers=2,
        max_seq_len=max_seq_len,
        seed=300,
    )
    model = TorchMiniLLM(source)
    model.eval()

    prompt = torch.tensor([[7, 11, 13]], dtype=torch.long)
    new_tokens = 12

    output = generate(model, prompt, new_tokens)

    # 2. Shape, dtype, and that the prompt survived unchanged at the front.
    assert output.shape == (1, prompt.shape[1] + new_tokens)
    assert output.dtype == torch.long
    assert torch.equal(output[:, :prompt.shape[1]], prompt)

    # The caller's tensor must not have been extended in place.
    assert prompt.shape == (1, 3)

    # 3. Every produced ID must be a valid index into the vocabulary.
    assert int(output.min()) >= 0
    assert int(output.max()) < vocab_size

    # 4. Greedy decoding is deterministic: same prompt, same continuation.
    repeated = generate(model, prompt, new_tokens)
    assert torch.equal(output, repeated)

    # 5. Generating past the context window must not raise. The model
    #    rejects sequences longer than max_seq_len, so this only passes
    #    if the crop in the loop is working.
    long_run = generate(model, prompt, max_seq_len + 10)
    assert long_run.shape == (1, prompt.shape[1] + max_seq_len + 10)

    # 6. A prompt that is ALREADY longer than the window must work too,
    #    which is what cropping before the forward pass buys us.
    over_long = torch.randint(
        0, vocab_size, (1, max_seq_len + 6), dtype=torch.long
    )
    extended = generate(model, over_long, 4)
    assert extended.shape == (1, max_seq_len + 10)
    assert torch.equal(extended[:, :max_seq_len + 6], over_long)

    # 7. Batches advance independently, each on its own prediction.
    batched = torch.tensor([[7, 11, 13], [21, 22, 23]], dtype=torch.long)
    batched_output = generate(model, batched, 5)
    assert batched_output.shape == (2, 8)

    # 8. Sampling is reproducible when the generator is seeded.
    first = generate(
        model,
        prompt,
        new_tokens,
        temperature=0.8,
        top_k=20,
        generator=torch.Generator().manual_seed(0),
    )
    second = generate(
        model,
        prompt,
        new_tokens,
        temperature=0.8,
        top_k=20,
        generator=torch.Generator().manual_seed(0),
    )
    assert torch.equal(first, second)

    # top_k=1 keeps only the best token, so sampling collapses onto greedy.
    forced = generate(
        model,
        prompt,
        new_tokens,
        temperature=1.0,
        top_k=1,
        generator=torch.Generator().manual_seed(1),
    )
    assert torch.equal(forced, output)

    # 9. Invalid inputs are rejected rather than silently reshaped.
    assert_rejects(
        lambda: generate(model, prompt[0], 4),
        "a 1-D prompt",
    )
    assert_rejects(
        lambda: generate(model, prompt.float(), 4),
        "a float prompt",
    )
    assert_rejects(
        lambda: generate(model, torch.zeros((1, 0), dtype=torch.long), 4),
        "an empty prompt",
    )
    assert_rejects(
        lambda: generate(model, prompt, 0),
        "a nonpositive token count",
    )
    assert_rejects(
        lambda: generate(model, prompt, 4, temperature=-1.0),
        "a negative temperature",
    )
    assert_rejects(
        lambda: generate(model, prompt, 4, top_k=0),
        "top_k below 1",
    )
    assert_rejects(
        lambda: generate(model, prompt, 4, top_k=vocab_size + 1),
        "top_k above the vocabulary size",
    )

    # 10. The model's mode is left exactly as it was found.
    model.train()
    generate(model, prompt, 2)
    assert model.training
    model.eval()

    print("Prompt IDs:", prompt.tolist())
    print("Greedy continuation:", output[0, prompt.shape[1]:].tolist())
    print("Sampled continuation:", first[0, prompt.shape[1]:].tolist())
    print("Context window:", model.max_seq_len)
    print("Longest run generated:", long_run.shape[1], "tokens")
    print("Over-long prompt handled:", over_long.shape[1], "->", extended.shape[1])
    print("Batched output:", tuple(batched_output.shape))
    print("Determinism, cropping, sampling, and rejection checks passed.")


if __name__ == "__main__":
    main()
