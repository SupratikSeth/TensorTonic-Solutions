def memory_accountant(
    param_shapes: list[list[int]], param_bytes_per_element: int,
    grad_bytes_per_element: int, activation_shapes: list[list[int]],
    activation_bytes_per_element: int, optimizer: str,
    optimizer_bytes_per_element: int,
) -> dict:
    """
    Returns integer byte counts for parameters, gradients, activations, optimizer_state, and total.
    """
    total_parameter = 0
    for shape in param_shapes:
        total_parameter += math.prod(shape)

    total_parameter_size = total_parameter * param_bytes_per_element
    total_gradient_size = total_parameter * grad_bytes_per_element

    total_activation = 0
    for shape in activation_shapes:
        total_activation += math.prod(shape)

    total_activation_size = total_activation * activation_bytes_per_element

    if optimizer == "sgd":
        total_optimizer_state_size = 0
    elif optimizer == "adagrad":
        total_optimizer_state_size = total_parameter * optimizer_bytes_per_element
    else:
        total_optimizer_state_size = 2 * total_parameter * optimizer_bytes_per_element

    total = total_parameter_size + total_gradient_size + total_activation_size + total_optimizer_state_size

    return {
        "parameters": total_parameter_size,
        "gradients": total_gradient_size,
        "activations": total_activation_size,
        "optimizer_state": total_optimizer_state_size,
        "total": total
    }