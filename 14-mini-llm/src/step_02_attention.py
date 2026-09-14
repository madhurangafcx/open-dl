import numpy as np


class AttentionHead:
    def __init__(
        self,
        d_model: int,
        head_dim: int,
        seed: int = 42,
    ) -> None:
        if d_model <= 0 or head_dim <= 0:
            raise ValueError("Dimensions must be positive")

        self.head_dim = head_dim
        rng = np.random.default_rng(seed)

        # Initialize once. The same matrices serve every forward pass.
        scale = 1.0 / np.sqrt(d_model)
        shape = (d_model, head_dim)

        self.Wq = rng.normal(0, scale, shape).astype(np.float32)
        self.Wk = rng.normal(0, scale, shape).astype(np.float32)
        self.Wv = rng.normal(0, scale, shape).astype(np.float32)

    def forward(
        self,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # x: (batch, sequence_length, d_model)
        sequence_length = x.shape[1]

        # 1. Project each input into query, key, and value vectors.
        q = x @ self.Wq
        k = x @ self.Wk
        v = x @ self.Wv

        # 2. Compare every query against every key.
        # Swap only the last two axes, preserving the batch axis.
        scores = q @ k.swapaxes(-2, -1)
        scores = scores / np.sqrt(self.head_dim)

        # 3. Block future positions. The diagonal remains visible.
        future_mask = np.triu(
            np.ones(
                (sequence_length, sequence_length),
                dtype=bool,
            ),
            k=1,
        )
        scores = np.where(
            future_mask[None, :, :],
            -np.inf,
            scores,
        )

        # 4. Numerically stable softmax over key positions.
        scores = scores - scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores)
        weights = exp_scores / exp_scores.sum(
            axis=-1, keepdims=True
        )

        # 5. Combine value vectors using those proportions.
        output = weights @ v

        return output, weights