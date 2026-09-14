import numpy as np

from step_03_multihead import MultiHeadAttention
from step_04_feedforward import FeedForward


class LayerNorm:
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
    ) -> None:
        if d_model <= 0 or eps <= 0:
            raise ValueError("d_model and eps must be positive")

        self.eps = eps

        # Learnable scale and shift, shared across positions.
        self.gamma = np.ones(d_model, dtype=np.float32)
        self.beta = np.zeros(d_model, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Reduce only the feature dimension.
        # (B, T, d_model) → (B, T, 1)
        mean = x.mean(axis=-1, keepdims=True)
        variance = x.var(axis=-1, keepdims=True)

        normalized = (x - mean) / np.sqrt(
            variance + self.eps
        )

        # Broadcasting applies one scale/shift per feature.
        return self.gamma * normalized + self.beta


class TransformerBlock:
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        seed: int = 200,
    ) -> None:
        # Attention and FFN have separate normalization parameters.
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            seed=seed,
        )

        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            seed=seed + num_heads + 1,
        )

    def forward(
        self,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # First branch: normalize, attend, then add the original input.
        normalized_x = self.norm1.forward(x)
        attention_output, weights = self.attention.forward(
            normalized_x
        )
        h = x + attention_output

        # Second branch: normalize, transform, then add h.
        normalized_h = self.norm2.forward(h)
        ffn_output = self.ffn.forward(normalized_h)
        output = h + ffn_output

        return output, weights