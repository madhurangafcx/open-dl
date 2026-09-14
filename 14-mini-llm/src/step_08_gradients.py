import numpy as np

from step_07_loss import cross_entropy


def loss_and_gradient(
    logits: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return mean cross-entropy and its gradient w.r.t. logits."""

    # Reuse the existing loss calculation and input validation.
    loss = cross_entropy(logits, targets)

    # Calculate stable vocabulary probabilities.
    shifted = logits - logits.max(
        axis=-1, keepdims=True
    )

    exp_scores = np.exp(shifted)
    probabilities = exp_scores / exp_scores.sum(
        axis=-1, keepdims=True
    )

    # Start with p, then subtract 1 at each correct token.
    gradient = probabilities.copy()

    batch_indices = np.arange(
        logits.shape[0]
    )[:, None]

    position_indices = np.arange(
        logits.shape[1]
    )[None, :]

    gradient[
        batch_indices,
        position_indices,
        targets,
    ] -= 1.0

    # The original loss is averaged over B × T positions.
    gradient /= targets.size

    return loss, gradient