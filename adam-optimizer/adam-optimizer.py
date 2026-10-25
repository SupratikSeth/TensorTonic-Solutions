import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    # Write code here
    param = np.array(param)
    grad = np.array(grad)
    m = np.array(m)
    v = np.array(v)
    print(param.shape, grad.shape, m.shape, v.shape)
    
    m_new = beta1 * m + (1 - beta1) * grad
    m_corrected = m_new / (1 - beta1**t)

    v_new = beta2 * v + ((1 - beta2) * grad**2)
    v_corrected = v_new / (1 - beta2**t)

    param = param - lr * (m_corrected / (np.sqrt(v_corrected) + eps))

    return (param, m_new, v_new)