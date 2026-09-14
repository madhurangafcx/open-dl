import numpy as np

from positions import positional_encoding
from step_05_transformer_block import (
    LayerNorm,
    TransformerBlock,
)


class MiniLLM:
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 64,
        seed: int = 300,
    ) -> None:
        if min(
            vocab_size,
            d_model,
            num_heads,
            num_layers,
            max_seq_len,
        ) <= 0:
            raise ValueError(
                "Model dimensions and counts must be positive"
            )

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        rng = np.random.default_rng(seed)

        # Trainable token embeddings: (V, d_model).
        self.token_embedding = rng.normal(
            loc=0.0,
            scale=0.02,
            size=(vocab_size, d_model),
        ).astype(np.float32)

        # Fixed positional information for the context window.
        self.positions = positional_encoding(
            max_seq_len, d_model
        )

        # Each layer is a separate block with its own parameters.
        self.blocks = [
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=4 * d_model,
                seed=seed + 1 + i * (num_heads + 2),
            )
            for i in range(num_layers)
        ]

        self.final_norm = LayerNorm(d_model)

        # Trainable output projection: (d_model, V).
        self.W_vocab = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(d_model),
            size=(d_model, vocab_size),
        ).astype(np.float32)

        self.b_vocab = np.zeros(
            vocab_size, dtype=np.float32
        )

    def forward(
        self,
        token_ids: np.ndarray,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        if token_ids.ndim != 2:
            raise ValueError(
                "Expected token IDs with shape (batch, sequence)"
            )

        if not np.issubdtype(token_ids.dtype, np.integer):
            raise ValueError("Token IDs must be integers")

        length = token_ids.shape[1]

        if not 1 <= length <= self.max_seq_len:
            raise ValueError(
                "Sequence length is outside the context window"
            )

        if (
            np.any(token_ids < 0)
            or np.any(token_ids >= self.vocab_size)
        ):
            raise ValueError(
                "Token ID is outside the vocabulary"
            )

        # Embed the IDs and add positions once.
        x = (
            self.token_embedding[token_ids]
            + self.positions[None, :length, :]
        )

        attention_maps = []

        # Each block processes the preceding block's output.
        for block in self.blocks:
            x, weights = block.forward(x)
            attention_maps.append(weights)

        # Normalize the final representations.
        x = self.final_norm.forward(x)

        # Produce one score per vocabulary token, per position.
        logits = x @ self.W_vocab + self.b_vocab

        return logits, attention_maps