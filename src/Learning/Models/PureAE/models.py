import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from .backbones import Encoder, Decoder, Decoder_medium, Decoder_vlarge

image_type = torch.Tensor

class Pure_SimpleAE(nn.Module):
    def __init__(self) -> None:
        super(Pure_SimpleAE, self).__init__()

        deconv_channels = [256, 128, 64] #assuming starting from 4x4 to 32x32
        self.encoder = Encoder()
        self.decoder = Decoder(deconv_channels)

    def forward(self, x: image_type):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Pure_SimpleAE_mediumDec(nn.Module):
    def __init__(self) -> None:
        super(Pure_SimpleAE_mediumDec, self).__init__()

        deconv_channels = [256, 128, 64] #assuming starting from 4x4 to 32x32
        self.encoder = Encoder()
        self.decoder = Decoder_medium(deconv_channels)

    def forward(self, x: image_type):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Pure_SimpleAE_vlargeDec(nn.Module):
    def __init__(self) -> None:
        super(Pure_SimpleAE_vlargeDec, self).__init__()

        deconv_channels = [256, 128, 64] #assuming starting from 4x4 to 32x32
        self.encoder = Encoder()
        self.decoder = Decoder_vlarge(deconv_channels)

    def forward(self, x: image_type):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

