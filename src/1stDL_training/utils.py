import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder


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

class get_FC_3layer(nn.Module):
    def __init__(self, bin):
        super().__init__()

        self.features = nn.Sequential(
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Linear(bin*4, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.features(x)
        return x

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.len = self.label.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx]).float()
        label = torch.tensor(self.label[idx]).long()
        return data, label

def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # fix the initial value of the network weight
    torch.cuda.manual_seed(seed)  # for cuda
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True  # choose the determintic algorithm


