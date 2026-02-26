import torch
import torch.nn as nn


class ZeroDCE(nn.Module):
    def __init__(self, num_iterations=8):
        super().__init__()
        self.num_iterations = num_iterations

        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(64, 32, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(64, 32, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(64, 3 * self.num_iterations, 3, 1, 1, bias=True)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        curves = x_r.chunk(self.num_iterations, dim=1)
        enhanced = x
        for curve in curves:
            enhanced = enhanced + curve * (torch.pow(enhanced, 2) - enhanced)

        enhanced = torch.clamp(enhanced, 0, 1)
        return enhanced
