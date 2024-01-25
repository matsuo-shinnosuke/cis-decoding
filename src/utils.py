import torch
import numpy as np
import pandas as pd
import torch.nn as nn


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

class CNN(nn.Module):
    def __init__(self, data_length=20, n_channel=50, last_dense=2):
        super().__init__()
        self.z_size = data_length
        for i in range(4):
            self.z_size = self.z_size//2
        self.features = nn.Sequential(
            nn.Conv1d(n_channel, 64, kernel_size=15, padding='same'),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=15, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
            nn.Conv1d(64, 32, kernel_size=10, padding='same'),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=10, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(32*self.z_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, last_dense),
        )

    def forward(self, x):
        x = x.transpose(2, 1)
        x = self.features(x)
        return x