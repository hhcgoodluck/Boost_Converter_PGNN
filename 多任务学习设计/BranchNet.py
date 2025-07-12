# branchnet.py

import torch.nn as nn

class BranchNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # 输出一个功率值
        )

    def forward(self, x):
        return self.model(x)
