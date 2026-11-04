import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    # Write code here
    if not y:
        return 0.0
    
    y = np.array(y)
    unique, count = np.unique(y, return_counts=True)
    p = count / len(y)
    mask = p > 0
    H = -np.sum(p[mask] * np.log2(p[mask]))

    return H