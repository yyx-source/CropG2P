import torch
import torch.nn as nn

class GatedInceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, use_residual=True):
        super().__init__()
        self.use_residual = use_residual

        # Ensure out_channels is divisible by 3
        assert out_channels % 3 == 0, f"out_channels ({out_channels}) must be divisible by 3"
        channels_per_branch = out_channels // 3

        # Branch 1: 1x1 convolution for local patterns
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, channels_per_branch, kernel_size=1),
            nn.BatchNorm1d(channels_per_branch, momentum=0.1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Branch 2: Standard 3x3 convolution for medium-scale patterns
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, channels_per_branch, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels_per_branch, momentum=0.1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Branch 3: Dilated 5x5 convolution for global patterns
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, channels_per_branch, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm1d(channels_per_branch, momentum=0.1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Lightweight gating with global average pooling and linear layer
        self.gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
            nn.Dropout(0.2)
        )

        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels,
                                  kernel_size=1) if use_residual and in_channels != out_channels else None

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        out = torch.cat([b1, b2, b3], dim=1)
        gate_weights = self.gate(out)
        out = out * gate_weights
        if self.use_residual and self.residual is not None:
            residual = self.residual(x)
            out = out + residual
        return out

class CropCNNMultiTaskModel(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        self.num_tasks = num_tasks
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 12, kernel_size=5, stride=1, padding=2),  # Reduced channels
            nn.BatchNorm1d(12, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            GatedInceptionModule(12, 48, use_residual=True),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(50),
        )

        self.fc1 = nn.Linear(48 * 50, 96)
        self.fc2 = nn.Linear(96, 32)
        self.fc_residual = nn.Linear(48 * 50, 32)

        assert isinstance(self.fc1, nn.Linear), "self.fc1 must be an nn.Linear layer"
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.4),
            self.fc2,
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(16, 1)
            ) for _ in range(num_tasks)
        ])

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.view(batch_size, -1)
        residual = self.fc_residual(x)
        x = self.fc1(x)
        x = self.fc(x) + residual
        return [head(x) for head in self.heads]
