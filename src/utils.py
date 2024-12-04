import torch
import numpy as np
import pandas as pd

def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # fix the initial value of the network weight
    torch.cuda.manual_seed(seed)  # for cuda
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True  # choose the determintic algorithm


def dna2onehot(x):
    dna2num_dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3,
                    'N': 3, 'n': 3, 'S': 3, 'K': 3,
                    'M': 3, 'W': 3, 'R': 3, 'Y': 3}
    x = pd.Series(x)
    x = x.map(dna2num_dict)

    # onehot
    x = np.eye(4)[x]
    return x