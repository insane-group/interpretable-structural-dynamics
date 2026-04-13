import torch.nn as nn

class RLInverseNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),   # outputs normalized R,L in [0,1]
        )

    def forward(self, x):
        return self.net(x)
