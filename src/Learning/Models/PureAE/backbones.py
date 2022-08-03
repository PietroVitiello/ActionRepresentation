from typing import List, Tuple
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = self.conv_layer(3, 64)
        self.conv2 = self.conv_layer(64, 128)
        self.conv3 = self.conv_layer(128, 192)
        self.conv4 = self.conv_layer(192, 256)
        self.conv5 = self.conv_layer(256, 256)

        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0, bias=True),
                                   nn.ReLU(inplace=True))

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

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class Decoder(nn.Module):
    def __init__(self, deconv_channels: List[int]) -> None:
        super(Decoder, self).__init__()

        self.deconv1 = self.deconv_layer(deconv_channels[0], deconv_channels[1])
        self.deconv2 = self.deconv_layer(deconv_channels[1], deconv_channels[2])
        self.deconv3 = self.deconv_layer(deconv_channels[2], 3)

    def deconv_layer(
            self,
            ch_in: int,
            ch_out: int,
            stride: int =2,
            kernel_size=3
        ):
        deconv = nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size, stride, padding=1, output_padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )
        return deconv

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x

class Decoder_medium(nn.Module):
    def __init__(self, deconv_channels: List[int]) -> None:
        super(Decoder_medium, self).__init__()

        self.deconv1 = self.deconv_layer(deconv_channels[0], deconv_channels[1])
        self.deconv2 = self.deconv_layer(deconv_channels[1], deconv_channels[1], stride=1, output_padding=0)
        self.deconv3 = self.deconv_layer(deconv_channels[1], deconv_channels[2])
        self.deconv4 = self.deconv_layer(deconv_channels[2], deconv_channels[2], stride=1, output_padding=0)
        self.deconv5 = self.deconv_layer(deconv_channels[2], 3)

    def deconv_layer(
            self,
            ch_in: int,
            ch_out: int,
            stride: int =2,
            kernel_size: int=3,
            output_padding: int=1
        ):
        deconv = nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size, stride, padding=1, output_padding=output_padding),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )
        return deconv

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        return x


class Decoder_vlarge(nn.Module):
    def __init__(self, deconv_channels: List[int]) -> None:
        super(Decoder_vlarge, self).__init__()

        self.deconv1 = self.deconv_layer(deconv_channels[0], deconv_channels[1], output_padding=0)
        self.deconv2 = self.deconv_layer(deconv_channels[1], deconv_channels[1])
        self.deconv3 = self.deconv_layer(deconv_channels[1], deconv_channels[2], output_padding=0)
        self.deconv4 = self.deconv_layer(deconv_channels[2], deconv_channels[2])
        self.deconv5 = self.deconv_layer(deconv_channels[2], 3)

    def deconv_layer(
            self,
            ch_in: int,
            ch_out: int,
            stride: int =2,
            kernel_size: int=3,
            output_padding: int=1
        ):
        deconv = nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size, stride, padding=1, output_padding=output_padding),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )
        return deconv

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        return x
        