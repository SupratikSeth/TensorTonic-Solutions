import numpy as np

def get_angles(positions: np.ndarray, dim: np.ndarray, d_model: int, base: float = 10000.0) -> np.ndarray:
    i = dim // 2
    angles = positions / (base ** (2 * i / d_model))
    return angles
    
def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    # Write code here
    positions = np.arange(seq_len).reshape((-1, 1))
    dim = np.arange(d_model).reshape((1, -1))

    angles = get_angles(positions, dim, d_model, base)

    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(angles[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angles[:, 1::2])

    return pos_encoding
    