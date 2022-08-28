import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from .activation_func import SpatialSoftArgmax

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
        bias=True,
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
