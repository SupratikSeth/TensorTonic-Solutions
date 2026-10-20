import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    m = X.shape[0]
    n = X.shape[1] 
 
    W = np.zeros(n)
    b = 0.0

    for step in range(steps):
        Z = np.dot(X, W) + b
        prob = _sigmoid(Z)

        dW = 1/m * np.dot(X.T, prob - y)
        db = 1/m * np.sum(prob - y)
        
        W = W - lr * dW
        b = b - lr * db

    return W, b