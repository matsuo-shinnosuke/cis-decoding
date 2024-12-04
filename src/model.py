import torch.nn as nn

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