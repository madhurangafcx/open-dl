import numpy as np

from step_02_attention import AttentionHead


class MultiHeadAttention:
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        seed: int = 42,
    ) -> None:
        if (
            d_model <= 0
            or num_heads <= 0
            or d_model % num_heads != 0
        ):
            raise ValueError(
                "d_model must be positive and divisible by num_heads"
            )

        self.head_dim = d_model // num_heads

        # Distinct initialization gives each head its own parameters.
        self.heads = [
            AttentionHead(
                d_model=d_model,
                head_dim=self.head_dim,
                seed=seed + i,
            )
            for i in range(num_heads)
        ]

        # Learnable output projection, initialized once.
        rng = np.random.default_rng(seed + num_heads)
        self.Wo = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(d_model),
            size=(d_model, d_model),
        ).astype(np.float32)

    def forward(
        self,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        head_outputs = []
        head_weights = []

        # Every head receives the same input.
        for head in self.heads:
            output, weights = head.forward(x)
            head_outputs.append(output)
            head_weights.append(weights)

        # Join features, preserving batch and sequence dimensions.
        # Four (B, T, 16) arrays become one (B, T, 64) array.
        combined = np.concatenate(head_outputs, axis=-1)

        # Mix the features contributed by the heads.
        output = combined @ self.Wo

        # Keep each head's attention matrix for inspection.
        # Four (B, T, T) arrays become (B, 4, T, T).
        weights = np.stack(head_weights, axis=1)

        return output, weights