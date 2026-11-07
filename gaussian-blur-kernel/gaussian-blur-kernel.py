import math

def gaussian_kernel(size: int, sigma: float) -> list:
    """
    Returns a square two-dimensional list.
    """
    # Write code here
    left_range = -(size // 2)
    right_range = size // 2

    G = []
    sum = 0.0
    for i in range(left_range, right_range + 1, 1):
        temp = []
        for j in range(left_range, right_range + 1, 1):
            weight = math.exp(-((i**2 + j**2) / (2 * sigma**2)))
            sum += weight
            temp.append(weight)
        G.append(temp)

    for i in range(size):
        for j in range(size):
            G[i][j] = G[i][j] / sum

    return G