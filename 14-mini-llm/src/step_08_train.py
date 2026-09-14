"""Step 8B: verify against NumPy, then overfit one fixed training batch.

From 14-mini-llm:
    ../.venv/bin/python src/step_08_train.py --device cpu
    ../.venv/bin/python src/step_08_train.py --device mps

Each run starts from the same NumPy initialization. The checkpoint contains
model weights and tokenizer settings for Step 9; optimizer state is not saved.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from bpe.tokenizer import Tokenizer
from step_06_language_model import MiniLLM
from step_08_gradients import loss_and_gradient
from step_08_torch_model import TorchMiniLLM


def token_loss(logits, targets):
    # (B,T,V) -> (B*T,V); targets (B,T) -> (B*T).
    # Targets are already shifted by the caller.
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
    )


def verify_numpy(source, model, inputs_np, targets_np):
    """Compare forward values and the first backward signal on CPU."""
    numpy_logits, _ = source.forward(inputs_np)
    numpy_loss, numpy_gradient = loss_and_gradient(numpy_logits, targets_np)
    inputs = torch.tensor(inputs_np, dtype=torch.long)
    targets = torch.tensor(targets_np, dtype=torch.long)

    logits = model(inputs)
    np.testing.assert_allclose(
        logits.detach().numpy(), numpy_logits, atol=3e-5, rtol=3e-5
    )
    loss = token_loss(logits, targets)
    np.testing.assert_allclose(loss.item(), numpy_loss, atol=1e-5, rtol=1e-5)

    # logits is an intermediate tensor, so explicitly retain its gradient.
    logits.retain_grad()
    loss.backward()
    np.testing.assert_allclose(
        logits.grad.numpy(), numpy_gradient, atol=1e-7, rtol=1e-4
    )
    # Ensure the backward graph reaches every trainable parameter.
    for name, value in model.named_parameters():
        if value.grad is None or not torch.isfinite(value.grad).all():
            raise AssertionError(f"Missing or nonfinite gradient: {name}")
    model.zero_grad(set_to_none=True)
    print("NumPy/PyTorch logits, loss, and loss-gradient checks passed.")
    print("Every model parameter received a finite gradient.")
    print(f"Initial loss: {loss.item():.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "step08",
    )
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable in this environment; use --device cpu")

    # A single CPU thread is sufficient for this very small exercise.
    torch.set_num_threads(1)
    tokenizer = Tokenizer()
    text = (
        "Transformers learn from sequences. "
        "Each position predicts the next token."
    )
    encoded = tokenizer.encode(text)
    ids = np.asarray(encoded.token_ids, dtype=np.int64)
    assert tokenizer.decode(ids.tolist()) == encoded.normalized_text
    assert set(tokenizer.vocabulary.id_to_token) == set(range(tokenizer.vocab_size))
    sequence_length = 16
    assert len(ids) >= sequence_length + 1
    inputs_np = ids[:sequence_length][None, :]
    targets_np = ids[1:sequence_length + 1][None, :]

    config = dict(
        vocab_size=tokenizer.vocab_size, d_model=64, num_heads=4,
        num_layers=2, max_seq_len=64, seed=300,
    )
    source = MiniLLM(**config)
    model = TorchMiniLLM(source)
    verify_numpy(source, model, inputs_np, targets_np)

    # Verification above is on CPU; training can use CPU or Apple's GPU.
    device = torch.device(args.device)
    model = model.to(device)
    inputs = torch.tensor(inputs_np, dtype=torch.long, device=device)
    targets = torch.tensor(targets_np, dtype=torch.long, device=device)
    initial_embedding = model.token_embedding.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    print(f"Training device: {device}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    model.train()
    for step in range(1, args.steps + 1):
        # Gradients accumulate unless cleared at the start of each update.
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = token_loss(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0, error_if_nonfinite=True
        )
        optimizer.step()

        if step == 1 or step % 25 == 0 or step == args.steps:
            # Measure AFTER the update, without building another graph.
            with torch.no_grad():
                measured_logits = model(inputs)
                measured_loss = token_loss(measured_logits, targets).item()
                accuracy = (measured_logits.argmax(dim=-1) == targets).float().mean().item()
            print(f"Step {step:3d}: loss={measured_loss:.6f}, accuracy={accuracy:.1%}")
            if measured_loss < 0.05 and accuracy == 1.0:
                break

    if not (measured_loss < 0.05 and accuracy == 1.0):
        raise RuntimeError(
            "The memorization target was not reached; rerun with --steps 800"
        )
    assert not torch.equal(initial_embedding, model.token_embedding.detach())
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.output_dir / "tokenizer")
    checkpoint_path = args.output_dir / "model.pt"
    # CPU tensors allow the checkpoint to be loaded on either CPU or MPS.
    torch.save({
        "config": config,
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "tokenizer_config": {
            "normalization_form": tokenizer.normalization_form,
            "pretokenizer_pattern": tokenizer.pretokenizer.pattern,
        },
        "training_steps": step,
        "training_loss": measured_loss,
        "training_text": text,
        "sequence_length": sequence_length,
    }, checkpoint_path)
    print("Embedding parameters changed. Tiny-batch memorization passed.")
    print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
