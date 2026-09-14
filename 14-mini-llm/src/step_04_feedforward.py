import numpy as np


def gelu(x: np.ndarray) -> np.ndarray:
    """Tanh approximation of the GELU activation."""
    scale = np.float32(np.sqrt(2.0 / np.pi))

    return 0.5 * x * (
        1.0 + np.tanh(
            scale * (x + 0.044715 * x**3)
        )
    )


class FeedForward:
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        seed: int = 100,
    ) -> None:
        if d_model <= 0 or d_ff <= 0:
            raise ValueError("Dimensions must be positive")

        rng = np.random.default_rng(seed)

        # Expand from model features to hidden features.
        self.W1 = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(d_model),
            size=(d_model, d_ff),
        ).astype(np.float32)

        self.b1 = np.zeros(d_ff, dtype=np.float32)

        # Project hidden features back to model features.
        self.W2 = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(d_ff),
            size=(d_ff, d_model),
        ).astype(np.float32)

        self.b2 = np.zeros(d_model, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # (B, T, 64) → (B, T, 256)
        hidden = x @ self.W1 + self.b1

        # Nonlinear transformation, preserving shape.
        hidden = gelu(hidden)

        # (B, T, 256) → (B, T, 64)
        output = hidden @ self.W2 + self.b2

        return output