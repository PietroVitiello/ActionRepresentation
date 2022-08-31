import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class SpatialSoftArgmax(nn.Module):
    def __init__(self) -> None:
        super(SpatialSoftArgmax, self).__init__()
        self.alpha = Parameter(torch.rand(1), requires_grad=True)

    def forward(self, x):
        x = torch.div(x, self.alpha)
        x = nn.Softmax2d()(x)
        x_x = torch.mean(torch.sum(x, dim=3), dim=2)
        x_y = torch.mean(torch.sum(x, dim=2), dim=2)
        return torch.cat((x_x, x_y), dim=1)

class SpatialSoftArgmax_strength(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spatialArgmax = SpatialSoftArgmax()
        self.activation = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.MaxPool2d(4, 1)
        )

    def forward(self, x):
        x_spatial = self.spatialArgmax(x)
        x_strength = torch.squeeze(
            self.activation(x),
            dim=-1
        )
        x_strength = torch.squeeze(x_strength, dim=-1)
        print(x_spatial.shape)
        print(x_strength.shape)
        # print(torch.cat((x_spatial, x_strength), dim=1).shape)
        return torch.cat((x_spatial, x_strength), dim=1)