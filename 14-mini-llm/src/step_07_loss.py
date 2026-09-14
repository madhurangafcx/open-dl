import numpy as np


def cross_entropy(
    logits: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Mean next-token loss for a batch without padding."""
    if (
        logits.ndim != 3
        or any(size == 0 for size in logits.shape)
    ):
        raise ValueError(
            "Expected nonempty logits with shape (B, T, V)"
        )

    if targets.shape != logits.shape[:2]:
        raise ValueError("Targets must have shape (B, T)")

    if not np.issubdtype(targets.dtype, np.integer):
        raise ValueError(
            "Targets must contain integer token IDs"
        )

    vocab_size = logits.shape[-1]

    if (
        np.any(targets < 0)
        or np.any(targets >= vocab_size)
    ):
        raise ValueError(
            "Target ID is outside the vocabulary"
        )

    if not np.isfinite(logits).all():
        raise ValueError("Logits must be finite")

    # Stable log-softmax across the vocabulary dimension.
    shifted = logits - logits.max(
        axis=-1, keepdims=True
    )

    log_normalizer = np.log(
        np.exp(shifted).sum(axis=-1, keepdims=True)
    )

    log_probs = shifted - log_normalizer

    # Select the correct token's log-probability
    # at every batch/sequence position.
    correct_log_probs = np.take_along_axis(
        log_probs,
        targets[..., None],
        axis=-1,
    ).squeeze(-1)

    # Average over all B × T prediction targets.
    return float(-correct_log_probs.mean())