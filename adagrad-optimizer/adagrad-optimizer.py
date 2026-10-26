import numpy as np

def adagrad_step(w: list, g: list, G: list, lr: float = 0.01, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with new_w and new_G.
    """
    # Write code here
    w = np.array(w)
    g = np.array(g)
    G = np.array(G)

    G_new = G + np.square(g)
    w_new = w - (lr / np.sqrt(G_new + eps)) * g

    return {
        "new_w": w_new,
        "new_G": G_new
    }