import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    # Write code here
    x_arr = np.array(x)
    return np.where(x_arr >= 0, 
                    1 / (1 + np.exp(-x_arr)), 
                    np.exp(x_arr) / (1 + np.exp(x_arr)))