import math

def cosine_annealing_schedule(base_lr: float, min_lr: float, total_steps: int, current_step: int) -> float:
    """
    Returns the cosine-annealed learning rate for the requested step.
    """
    # Write code here
    lr = min_lr + (1/2 * (base_lr - min_lr) * (1 + math.cos(current_step / total_steps * math.pi)))
    return lr