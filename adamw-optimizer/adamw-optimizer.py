import numpy as np

def adamw_step(w: list, m: list, v: list, grad: list, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, weight_decay: float = 0.01, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with new_w, new_m, and new_v.
    """
    # Write code here
    w = np.array(w)
    m = np.array(m)
    v = np.array(v)
    grad = np.array(grad)

    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * np.square(grad)

    w_new = w - (lr * weight_decay * w) - (lr * (m_new / (np.sqrt(v_new) + eps)))

    return {
        "new_w": w_new,
        "new_m": m_new,
        "new_v": v_new
    }