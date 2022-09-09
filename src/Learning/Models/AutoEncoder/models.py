import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from .backbones import SpatialAE_sceneUnderstanding_backbone
from .activation_func import SpatialSoftArgmax, SpatialSoftArgmax_strength

class SpatialAE(nn.Module):
    def __init__(
        self,
        num_outputs=6,
        num_aux_outputs=9,
        reconstruction_size=32
    ):
        super(SpatialAE,self).__init__()
        self.recon_size = reconstruction_size
        self.spatialArgmax = SpatialSoftArgmax()

        self.encoder = self.Encoder()
        self.decoder = self.Decoder(reconstruction_size)

        self.reach_fc = nn.Sequential(nn.Linear(256, 128),
                                 nn.ReLU())

        self.aux = nn.Linear(128, num_aux_outputs)
        self.out = nn.Linear(128, num_outputs)

        self.stop = nn.Sequential(nn.Linear(256, 64),
                                  nn.ReLU(),
                                  nn.Linear(64, 1),
                                  nn.Sigmoid())

    def Encoder(self):
        encoder = nn.Sequential(
            self.conv_layer(3, 32),
            self.conv_layer(32, 64),
            self.conv_layer(64, 128),
            self.spatialArgmax
        )
        return encoder

    def Decoder(self, img_size: int):
        decoder = nn.Linear(256, img_size**2)
        return decoder

    def decoder_forward(self, x: torch.Tensor):
        x = self.decoder(x)
        return x.view(x.size(0), 1, self.recon_size, self.recon_size)

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
        pool_padding=1,
        inplace=True
    ):
        conv = nn.Sequential(nn.Conv2d(chIN, chOUT, kernel_size, stride, padding, bias=bias),
                             nn.BatchNorm2d(chOUT),
                             nn.MaxPool2d(pool_kernel, pool_stride, pool_padding),
                             nn.ReLU(inplace))
        return conv

    def freeze_backbone(self):
        self.encoder.requires_grad_(False)
        self.spatialArgmax.requires_grad_(False)
        self.decoder.requires_grad_(False)
        self.reach_fc.requires_grad_(False)
        self.out.requires_grad_(False)
        self.aux.requires_grad_(False)

    def forward(self, x, train_stop: bool=None):
        x = self.encoder(x)

        if self.training:
            if train_stop == False:
                recon_img = self.decoder_forward(x)
                fc1 = self.reach_fc(x)
                x_aux = self.aux(fc1)
                x_out = self.out(fc1)
                out = torch.cat((x_out, x_aux), dim=1)
                return (out, recon_img)
            elif train_stop == True:
                return self.stop(x)
        else:
            reach = self.reach_fc(x)
            reach = self.out(reach)
            stop = self.stop(x)
            return (reach, stop)

class SpatialAE_fc(nn.Module):
    def __init__(
        self,
        num_outputs=6,
        num_aux_outputs=9,
        reconstruction_size=16
    ):
        super(SpatialAE_fc,self).__init__()
        self.recon_size = reconstruction_size
        self.spatialArgmax = SpatialSoftArgmax()

        self.encoder = self.Encoder()
        self.decoder = self.Decoder(reconstruction_size)

        self.fc1 = nn.Sequential(nn.Linear(256, 128),
                                 nn.ReLU())

        self.aux = nn.Linear(128, num_aux_outputs)
        self.out = nn.Linear(128, num_outputs)

    def Encoder(self):
        encoder = nn.Sequential(
            self.conv_layer(3, 32),
            self.conv_layer(32, 64),
            self.conv_layer(64, 128),
            self.spatialArgmax
        )
        return encoder

    def Decoder(self, img_size: int):
        decoder = nn.Linear(256, img_size**2)
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

    def forward(self, x):
        x: torch.Tensor = self.encoder(x)     
        fc1 = self.fc1(x)

        if self.training:
            recon_img = self.decoder_forward(x)
            x_aux = self.aux(fc1)
            x_out = self.out(fc1)
            out = torch.cat((x_out, x_aux), dim=1)
            return (recon_img, out)
        else:
            return self.out(fc1)


class StrengthSpatialAE_fc(nn.Module):
    def __init__(
        self,
        num_outputs=6,
        num_aux_outputs=9,
        reconstruction_size=16
    ):
        super(StrengthSpatialAE_fc,self).__init__()
        self.recon_size = reconstruction_size
        self.spatialArgmax = SpatialSoftArgmax_strength()

        self.encoder = self.Encoder()
        self.decoder = self.Decoder()

        self.fc1 = nn.Sequential(nn.Linear(384, 128),
                                 nn.ReLU())

        self.aux = nn.Linear(128, num_aux_outputs)
        self.out = nn.Linear(128, num_outputs)

    def Encoder(self):
        encoder = nn.Sequential(
            self.conv_layer(3, 32),
            self.conv_layer(32, 64),
            self.conv_layer(64, 128),
            self.spatialArgmax
        )
        return encoder

    def Decoder(self):
        decoder = nn.Linear(384, self.recon_size**2)
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

    def forward(self, x):
        # print(x.shape)
        x: torch.Tensor = self.encoder(x) 
        # print(f"x after activation: {x.shape}")       
        fc1 = self.fc1(x)

        if self.training:
            recon_img = self.decoder_forward(x)
            x_aux = self.aux(fc1)
            x_out = self.out(fc1)
            out = torch.cat((x_out, x_aux), dim=1)
            return (recon_img, out)
        else:
            return self.out(fc1)


# class SpatialAE_CNN(nn.Module):
#     def __init__(
#         self,
#         num_outputs=6,
#         num_aux_outputs=9,
#         reconstruction_size=16
#     ):
#         super(StrengthSpatialAE_fc,self).__init__()
#         self.recon_size = reconstruction_size
#         self.spatialArgmax = SpatialSoftArgmax_strength()

#         self.encoder = self.Encoder()
#         self.decoder = self.Decoder(reconstruction_size)

#         self.fc1 = nn.Sequential(nn.Linear(256, 128),
#                                  nn.ReLU())

#         self.aux = nn.Linear(128, num_aux_outputs)
#         self.out = nn.Linear(128, num_outputs)

#     def Encoder(self):
#         encoder = nn.Sequential(
#             self.conv_layer(3, 32),
#             self.conv_layer(32, 64),
#             self.conv_layer(64, 128),
#             self.spatialArgmax
#         )
#         return encoder

#     def Decoder(self, img_size: int):
#         decoder = nn.Linear(256, img_size**2)
#         return decoder

#     def decoder_forward(self, x: torch.Tensor):
#         self.decoder(x)
#         return x.view(x.size(0), self.recon_size, -1)

#     def conv_layer(
#         self,
#         chIN,
#         chOUT,
#         kernel_size=3,
#         stride=1,
#         padding=1,
#         bias=False,
#         pool_kernel=3,
#         pool_stride=2,
#         pool_padding=1
#     ):
#         conv = nn.Sequential(nn.Conv2d(chIN, chOUT, kernel_size, stride, padding, bias=bias),
#                              nn.BatchNorm2d(chOUT),
#                              nn.MaxPool2d(pool_kernel, pool_stride, pool_padding),
#                              nn.ReLU(inplace=True))
#         return conv

#     def forward(self, x):
#         # print(x.shape)
#         x: torch.Tensor = self.encoder(x) 
#         # print(f"x after activation: {x.shape}")       
#         fc1 = self.fc1(x)

#         if self.training:
#             recon_img = self.decoder_forward(x)
#             x_aux = self.aux(fc1)
#             x_out = self.out(fc1)
#             out = torch.cat((x_out, x_aux), dim=1)
#             return (recon_img, out)
#         else:
#             return self.out(fc1)

class SpatialAE_sceneUnderstanding(nn.Module):
    def __init__(
        self,
        num_outputs=6,
        num_aux_outputs=9,
        reconstruction_size=16
    ):
        super(SpatialAE_sceneUnderstanding,self).__init__()

        self.backbone = SpatialAE_sceneUnderstanding_backbone(reconstruction_size)

        # Reach
        self.reach_fc = nn.Sequential(nn.Linear(128+128+5, 128), #in: end_cnn + spatial + cubePos
                                      nn.ReLU())
        self.reach_lstm = nn.LSTM(
                            input_size = 128+6, #previous layer + previous 6dof action
                            hidden_size= 64,
                            num_layers= 1,
                            batch_first=True
                        )
        self.aux = nn.Linear(64, num_aux_outputs)
        self.out = nn.Linear(64, num_outputs)

        # Stop
        self.stop = nn.Sequential(nn.Linear(128+128+5, 64),
                                  nn.ReLU(),
                                  nn.Linear(64, 1),
                                  nn.Sigmoid())

        self.h, self.c = None, None

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

    def forward(self, x, eePos, prevAction, cubePos, relativeCubePos, train_stop=None):
        if self.training:
            if train_stop == False:
                x, spatial, cube_pos_cartesian, cube_pos_relative, recon_img = self.backbone(x, eePos, train_stop=False)
                x = torch.cat((x, spatial, cubePos, relativeCubePos), dim=1)
                x = self.reach_fc(x)
                x = torch.cat((x, prevAction), dim=1)
                x = x.view(1, x.size(0),-1)
                x = self.reach_lstm(x)
                x = x.squeeze()
                x_out = self.out(x)
                x_aux = self.aux(x)
                return (x_out, x_aux, cube_pos_cartesian, cube_pos_relative)

            elif train_stop == True:
                with torch.no_grad():
                    x, spatial, _, _, _ = self.backbone(x, eePos, train_stop=True)
                x = torch.cat((x, spatial, cubePos, relativeCubePos), dim=1)
                stop_signal = self.stop(x)
                return (stop_signal)            
        else:
            x, spatial, cube_pos_cartesian, cube_pos_relative, _ = self.backbone(x, eePos)
            x = torch.cat((x, spatial, cube_pos_cartesian, cube_pos_relative), dim=1)

            reach = self.reach_fc(x)
            reach = torch.cat((reach, prevAction), dim=1)
            reach = reach.view(1, reach.size(0),-1)
            reach, (self.h, self.c) = self.reach_lstm(reach, (self.h, self.c))
            reach = reach.squeeze()
            reach = self.out(reach)

            stop_signal = self.stop(x)
            return (reach, stop_signal)