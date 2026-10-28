def linear_lr(step: int, total_steps: int, initial_lr: float, final_lr: float = 0.0, warmup_steps: int = 0) -> float:
    """
    Returns the learning rate as a float.
    """
    # Write code here
    if step < warmup_steps:
        lr = initial_lr * (step / warmup_steps)
    elif step >= warmup_steps and step < total_steps:
        lr = final_lr + ((initial_lr - final_lr) * ((total_steps - step) / (total_steps - warmup_steps)))
    else:
        lr = final_lr

    return lr