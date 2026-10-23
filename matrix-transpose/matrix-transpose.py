import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    m = len(A)
    n = len(A[0])

    transpose = np.zeros(shape=(n, m))
    for i in range(m):
        for j in range(n):
            transpose[j, i] = A[i][j]

    return transpose
