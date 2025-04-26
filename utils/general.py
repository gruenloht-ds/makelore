import random
import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
        
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False