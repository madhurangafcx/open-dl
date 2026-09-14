"""Step 10: load the trained checkpoint and expose a text-in, text-out API.

From 14-mini-llm:
    ../.venv/bin/python src/step_10_app.py
    ../.venv/bin/python src/step_10_app.py --device mps --prompt "Transformers"

Step 8 already wrote the checkpoint, so this module supplies the other
half: reading it back, proving the weights really landed, and wrapping
the result so a caller never has to know MiniLLM, TorchMiniLLM, or
Tokenizer exist.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from bpe.pretokenizer import PreTokenizer
from bpe.tokenizer import Tokenizer
from step_06_language_model import MiniLLM
from step_08_torch_model import TorchMiniLLM
from step_08_train import token_loss
from step_09_generate import generate

DEFAULT_ARTIFACTS = (
    Path(__file__).resolve().parents[1] / "artifacts" / "step08"
)


def load_model(checkpoint, device):
    """Rebuild the trained model from a checkpoint's config and weights.

    TorchMiniLLM copies its shapes from a NumPy MiniLLM, so one is built
    here purely to be measured -- its random weights are overwritten by
    load_state_dict immediately. The detour is deliberate: it keeps a
    single definition of the architecture instead of restating every
    shape a second time.
    """
    source = MiniLLM(**checkpoint["config"])
    model = TorchMiniLLM(source)

    # strict=True by default. `positions` is a registered buffer and so
    # appears in the state dict -- a mismatch there is a real problem,
    # not something to wave through.
    report = model.load_state_dict(checkpoint["model_state"])

    if report.missing_keys or report.unexpected_keys:
        raise RuntimeError(
            "Checkpoint does not match the model: "
            f"missing={report.missing_keys}, "
            f"unexpected={report.unexpected_keys}"
        )

    model.to(device)
    model.eval()

    return model


def load_tokenizer(directory, tokenizer_config):
    """Rebuild the tokenizer, including the two settings save() drops.

    Tokenizer.save writes vocab.json and merges.txt only. The
    normalization form and pre-tokenizer pattern are not persisted, and
    a mismatch fails silently -- merges stop applying and token counts
    quietly grow. Step 8 stored both in the checkpoint for this reason.
    """
    return Tokenizer.load(
        directory,
        normalization_form=tokenizer_config["normalization_form"],
        pretokenizer=PreTokenizer(
            tokenizer_config["pretokenizer_pattern"]
        ),
    )


class MiniLLMApp:
    """The application-facing surface: a directory in, strings out."""

    def __init__(self, artifacts_dir=DEFAULT_ARTIFACTS, device: str = "cpu"):
        artifacts_dir = Path(artifacts_dir)
        checkpoint_path = artifacts_dir / "model.pt"
        tokenizer_dir = artifacts_dir / "tokenizer"

        # A bare FileNotFoundError from torch.load names the file but not
        # the fix, and "run step 8" is the only useful thing to say here.
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"No checkpoint at {checkpoint_path}. "
                "Train one first: python src/step_08_train.py"
            )

        if not tokenizer_dir.is_dir():
            raise FileNotFoundError(
                f"No tokenizer directory at {tokenizer_dir}. "
                "Train one first: python src/step_08_train.py"
            )

        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS is unavailable in this environment; use device='cpu'"
            )

        self.device = torch.device(device)

        # The checkpoint holds tensors, ints, floats, and strings only,
        # so it loads without unpickling arbitrary objects. It was saved
        # as CPU tensors; map_location keeps that true before .to(device).
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )

        self.config = checkpoint["config"]
        self.metadata = {
            key: checkpoint[key]
            for key in (
                "training_steps",
                "training_loss",
                "training_text",
                "sequence_length",
            )
        }

        self.model = load_model(checkpoint, self.device)
        self.tokenizer = load_tokenizer(
            tokenizer_dir, checkpoint["tokenizer_config"]
        )

    def verify(self) -> None:
        """Prove the load worked before anyone depends on the answers."""
        # 1. A tokenizer and an embedding table from different runs would
        #    index each other incorrectly without ever raising.
        if self.tokenizer.vocab_size != self.config["vocab_size"]:
            raise RuntimeError(
                "Tokenizer vocabulary does not match the model: "
                f"{self.tokenizer.vocab_size} vs "
                f"{self.config['vocab_size']}"
            )

        text = self.metadata["training_text"]
        encoded = self.tokenizer.encode(text)

        # 2. The same round-trip Steps 1 and 8 assert, on the saved text.
        assert (
            self.tokenizer.decode(encoded.token_ids)
            == encoded.normalized_text
        )

        # 3. The strong check. Matching shapes prove nothing about the
        #    values, but reproducing the final training loss means every
        #    weight landed where it was saved from.
        ids = np.asarray(encoded.token_ids, dtype=np.int64)
        length = self.metadata["sequence_length"]

        inputs = torch.tensor(
            ids[:length][None, :], dtype=torch.long, device=self.device
        )
        targets = torch.tensor(
            ids[1:length + 1][None, :],
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            loss = token_loss(self.model(inputs), targets).item()

        np.testing.assert_allclose(
            loss, self.metadata["training_loss"], atol=1e-4, rtol=1e-3
        )

    def complete(
        self,
        prompt: str,
        max_new_tokens: int = 16,
        *,
        temperature: float = 0.0,
        top_k: int | None = None,
        include_prompt: bool = True,
        generator=None,
    ) -> str:
        """Continue `prompt` and return the text.

        Returns the prompt plus its continuation by default; set
        include_prompt=False for the continuation alone.

        The result is *normalized* text, because decode() reverses the
        byte encoding but not the Unicode normalization applied on the
        way in. For text that normalization does not alter -- which is
        most text -- the two are identical.

        A `generator` makes sampling repeatable and must live on the same
        device as the model.
        """
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string")

        prompt_ids = self.tokenizer.encode(prompt).token_ids

        # The model requires at least one position; an empty prompt would
        # otherwise fail deep inside the forward pass.
        if not prompt_ids:
            raise ValueError("Prompt must contain at least one token")

        token_ids = torch.tensor(
            [prompt_ids], dtype=torch.long, device=self.device
        )

        output = generate(
            self.model,
            token_ids,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            generator=generator,
        )

        produced = output[0].tolist()

        if not include_prompt:
            produced = produced[len(prompt_ids):]

        return self.tokenizer.decode(produced)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS
    )
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=15)
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

    torch.set_num_threads(1)

    app = MiniLLMApp(
        artifacts_dir=args.artifacts_dir, device=args.device
    )
    app.verify()

    text = app.metadata["training_text"]
    length = app.metadata["sequence_length"]
    ids = app.tokenizer.encode(text).token_ids

    # Step 8 drove the loss to near zero on one 16-token window, so greedy
    # decoding from its first token must reproduce that window exactly.
    # A fault in the crop, the last-position slice, or the append all
    # break this, which is what makes it worth asserting.
    seed_ids = torch.tensor(
        [ids[:1]], dtype=torch.long, device=app.device
    )
    recalled = generate(app.model, seed_ids, length - 1)
    assert recalled[0].tolist() == ids[:length]

    # The same claim, made through the public API instead of tensors.
    expected_text = app.tokenizer.decode(ids[:length])
    produced_text = app.complete(
        app.tokenizer.decode(ids[:1]), length - 1
    )
    assert produced_text == expected_text

    print("Checkpoint:", args.artifacts_dir / "model.pt")
    print("Vocabulary size:", app.tokenizer.vocab_size)
    print("Device:", app.device)
    print("Training steps:", app.metadata["training_steps"])
    print(f"Training loss: {app.metadata['training_loss']:.6f}")
    print("Load verification passed: vocabulary, round-trip, and loss.")
    print(f"Memorized window reproduced: {produced_text!r}")

    prompt = args.prompt if args.prompt is not None else text[:12]
    continuation = app.complete(
        prompt, args.max_new_tokens, include_prompt=False
    )

    print(f"Prompt: {prompt!r}")
    print(f"Continuation: {continuation!r}")


if __name__ == "__main__":
    main()
