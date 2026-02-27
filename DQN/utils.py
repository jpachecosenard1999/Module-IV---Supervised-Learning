import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def linear_epsilon(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    """Decaimiento lineal de epsilon por pasos."""
    if step >= decay_steps:
        return eps_end
    frac = step / decay_steps
    return eps_start + frac * (eps_end - eps_start)
