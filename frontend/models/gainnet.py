import torch
import torch.nn as nn
import torch.nn.functional as F


class GainNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(3,32,3,padding=1)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.conv3 = nn.Conv2d(64,64,3,padding=1)
        self.conv4 = nn.Conv2d(64,32,3,padding=1)
        self.conv5 = nn.Conv2d(32,3,3,padding=1)

    def forward(self,x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        gain = F.softplus(
            self.conv5(x)
        )

        return gain


def enhance(low,gain):

    enhanced = low * gain

    enhanced = torch.clamp(
        enhanced,
        0,
        1
    )

    return enhanced