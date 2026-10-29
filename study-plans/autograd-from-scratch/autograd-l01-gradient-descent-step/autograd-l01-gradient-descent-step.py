import numpy as np

def gradient_descent_step(values: list, gradients: list, learning_rate: float) -> tuple[list, float]:
    """
    Returns a fresh list of updated values and the predicted objective change.
    """
    weights = np.array(values, dtype=np.float64)
    grads = np.array(gradients, dtype=np.float64)

    weights_updated = weights - learning_rate * grads
    predicted_loss_change = np.sum(grads * (weights_updated - weights))

    return [float(weight) for weight in weights_updated], float(predicted_loss_change)