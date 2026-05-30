import torch
import torch.nn as nn
import torch.nn.functional as F


class DCENet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            3,32,3,padding=1
        )

        self.conv2 = nn.Conv2d(
            32,32,3,padding=1
        )

        self.conv3 = nn.Conv2d(
            32,32,3,padding=1
        )

        self.conv4 = nn.Conv2d(
            32,32,3,padding=1
        )

        self.conv5 = nn.Conv2d(
            64,32,3,padding=1
        )

        self.conv6 = nn.Conv2d(
            64,32,3,padding=1
        )

        self.conv7 = nn.Conv2d(
            64,24,3,padding=1
        )

    def forward(self,x):

        x1 = F.relu(
            self.conv1(x)
        )

        x2 = F.relu(
            self.conv2(x1)
        )

        x3 = F.relu(
            self.conv3(x2)
        )

        x4 = F.relu(
            self.conv4(x3)
        )

        x5 = F.relu(
            self.conv5(
                torch.cat(
                    [x3,x4],
                    dim=1
                )
            )
        )

        x6 = F.relu(
            self.conv6(
                torch.cat(
                    [x2,x5],
                    dim=1
                )
            )
        )

        out = torch.tanh(
            self.conv7(
                torch.cat(
                    [x1,x6],
                    dim=1
                )
            )
        )

        return out


def enhance(
    image,
    maps
):

    out = image

    for i in range(8):

        A = maps[
            :,
            3*i:3*i+3,
            :,
            :
        ]

        out = (
            out
            +
            A * out * (1-out)
        )

    return out