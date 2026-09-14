import numpy as np

def positional_encoding(
    sequence_length: int,
    d_model: int,
) -> np.ndarray:
    """Create fixed position vectors with shape (T, d_model)."""

    if d_model <= 0 or d_model % 2 != 0:
        raise ValueError("d_model must be a positive even number")

    # One position per row: shape (T, 1).
    positions = np.arange(
        sequence_length, dtype=np.float32
    )[:, None]

    # Even dimensions: 0, 2, 4, ...
    even_dimensions = np.arange(
        0, d_model, 2, dtype=np.float32
    )

    # Broadcasting produces shape (T, d_model / 2).
    angles = positions / (
        10000.0 ** (even_dimensions / d_model)
    )

    encoding = np.zeros(
        (sequence_length, d_model), dtype=np.float32
    )

    encoding[:, 0::2] = np.sin(angles)
    encoding[:, 1::2] = np.cos(angles)

    return encoding