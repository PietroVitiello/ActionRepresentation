import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from .backbones import BaselineCNN_backbone

class Stop_AuxBaselineCNN(nn.Module):
    def __init__(self, num_outputs=6, num_aux_outputs=9):
        super(Stop_AuxBaselineCNN, self).__init__()

        self.cnn_backbone = BaselineCNN_backbone()

        self.reach_fc = nn.Sequential(nn.Linear(256, 128),
                                      nn.ReLU())

        self.aux = nn.Linear(128, num_aux_outputs)
        self.out = nn.Linear(128, num_outputs)

        self.stop = nn.Sequential(nn.Linear(256, 64),
                                  nn.ReLU(),
                                  nn.Linear(64, 1),
                                  nn.Sigmoid())

    def freeze_backbone(self):
        self.cnn_backbone.requires_grad_(False)
        self.reach_fc.requires_grad_(False)
        self.out.requires_grad_(False)
        self.aux.requires_grad_(False)

    def forward(self, x, train_stop: bool=None):
        if self.training:
            if train_stop == False:
                x = self.cnn_backbone(x)
                x = self.reach_fc(x)
                x_aux = self.aux(x)
                x_out = self.out(x)
                return torch.cat((x_out, x_aux), dim=1)
            if train_stop == True:
                # with torch.no_grad():
                x: torch.Tensor = self.cnn_backbone(x)
                # x.requires_grad = True
                return self.stop(x)
        else:
            x = self.cnn_backbone(x)

            reach_x = self.reach_fc(x)
            reach_x = self.out(reach_x)
            stop_signal = self.stop(x)
            return (reach_x, stop_signal)


class SpatialSoftArgmax(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = Parameter(torch.rand(1), requires_grad=True)

    def forward(self, x):
        # print(f"x inside activation: {x.shape}")
        x = torch.div(x, self.alpha)
        # print(f"x after division: {x.shape}")
        x = nn.Softmax2d()(x)
        # print(f"x after softmax: {x.shape}")
        x_x = torch.mean(torch.sum(x, dim=3), dim=2)
        x_y = torch.mean(torch.sum(x, dim=2), dim=2)
        # print(f"x_x: {x_x.shape}")
        # print(f"x_y: {x_y.shape}")
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

class SpatialAE_sceneUnderstanding_backbone(nn.Module):
    def __init__(
        self,
        reconstruction_size=16
    ):
        super(SpatialAE_sceneUnderstanding_backbone,self).__init__()

        # Initial CNN
        self.encoder = self.Encoder()

        # AutoEncoder
        self.recon_size = reconstruction_size
        self.spatialArgmax = SpatialSoftArgmax()
        self.decoder = self.Decoder(reconstruction_size)

        # End CNN
        self.end_cnn = nn.Sequential(
            self.conv_layer(64, 128),
            self.conv_layer(128, 128),
            nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )

        # Memory of cube position
        self.cubePos_lstm = nn.LSTM(
                            input_size = 128+6, #previous layer + 6dof ee position
                            hidden_size= 64,
                            num_layers= 1,
                            batch_first=True
                        )
        self.cubePos = nn.Linear(64, 5) # cartesian cube pos + up/down + left/right

        self.h, self.c = None, None

    def Encoder(self):
        encoder = nn.Sequential(
            self.conv_layer(3, 32),
            self.conv_layer(32, 64),
            self.conv_layer(64, 64) #might want to change this
        )
        return encoder

    def Decoder(self, img_size: int):
        decoder = nn.Linear(128, img_size**2)
        return decoder

    def decoder_forward(self, x: torch.Tensor):
        x = self.decoder(x)
        return x.view(x.size(0), self.recon_size, -1)

    def conv_layer(
        self,
        chIN,
        chOUT,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        pool_kernel=3,
        pool_stride=2,
        pool_padding=1
    ):
        conv = nn.Sequential(nn.Conv2d(chIN, chOUT, kernel_size, stride, padding, bias=bias),
                             nn.BatchNorm2d(chOUT),
                             nn.MaxPool2d(pool_kernel, pool_stride, pool_padding),
                             nn.ReLU(inplace=True))
        return conv

    def forward(self, x: torch.Tensor, eePos: torch.Tensor, train_stop: bool=False):
        x = self.encoder(x) 
        spatial = self.spatialArgmax(x)

        x = self.end_cnn(x)
        x = x.view(x.size(0),-1)
        recon_img = None

        if train_stop:
            cube_pos_cartesian = None
            cube_pos_relative = None
        else:
            if self.training:
                recon_img = self.decoder_forward(spatial)

                cube_pos = x.view(1, x.size(0),-1)
                cube_pos = torch.cat((x, eePos), dim=1)
                cube_pos, _ = self.cubePos_lstm(cube_pos)
                cube_pos = cube_pos.squeeze()
            else:
                cube_pos = x.view(x.size(0),-1)
                cube_pos = torch.cat((x, eePos), dim=1)
                cube_pos, (self.h, self.c) = self.cubePos_lstm(cube_pos, (self.h, self.c))
                cube_pos = cube_pos.squeeze()

            cube_pos = self.cubePos(cube_pos)
            cube_pos_cartesian = cube_pos[:,:3]
            cube_pos_relative = nn.Sigmoid()(cube_pos[:,3:])

        return (x, spatial, cube_pos_cartesian, cube_pos_relative, recon_img)
