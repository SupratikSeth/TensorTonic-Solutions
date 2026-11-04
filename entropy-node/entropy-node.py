import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    # Write code here
    if not y:
        return 0.0
    
    y = np.array(y)
    unique = np.unique(y)
    p = np.array([np.count_nonzero(y == cls) / len(y) for cls in unique])
    mask = p > 0
    H = -1 * np.sum(p[mask] * np.log2(p[mask]))

    return H