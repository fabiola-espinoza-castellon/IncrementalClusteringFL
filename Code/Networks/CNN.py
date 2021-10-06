import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNetwork(nn.Module):

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=5, bias=True, padding=5//2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, bias=True, padding=5//2)

        self.fc1 = nn.Linear(in_features=64*7*7, out_features=512, bias=True)
        self.out = nn.Linear(in_features=512, out_features=output_channels, bias=True)

    def forward(self, t):
        # (1) input layer
        t = t

        # (2) hidden conv layer
        t = self.conv1(t)
        t = torch.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) hidden conv layer
        t = self.conv2(t)
        t = torch.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (4) hidden linear layer
        t = t.reshape(-1, 64*7*7)
        t = self.fc1(t)
        t = torch.relu(t)

        # (5) output layer
        t = self.out(t)

        return t


