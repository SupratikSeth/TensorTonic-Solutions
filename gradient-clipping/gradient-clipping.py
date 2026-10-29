import numpy as np

def clip_gradients(g: list, max_norm: float) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as g.
    """
    # Write code here
    g = np.array(g)
    l2_norm = np.sqrt(np.sum(np.square(g)))

    if l2_norm <= max_norm:
        clipped_gradient = g
    else:
        scale = max_norm / l2_norm
        clipped_gradient = g * scale

    return clipped_gradient