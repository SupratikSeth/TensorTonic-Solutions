import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    # Write code here
    x = np.array(x)
    random = rng.random((x.shape))
    scaled_mask = (random > p) * (1 / (1 - p))
    output = x * scaled_mask

    return output, scaled_mask