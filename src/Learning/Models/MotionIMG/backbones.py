from typing import List, Tuple
import torch
import torch.nn as nn


class BaselineCNN_backbone(nn.Module):
    def __init__(self):
        super(BaselineCNN_backbone, self).__init__()

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
                             nn.ReLU(inplace=False))
        return conv

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_4x4 = self.conv4(x)
        x = self.conv5(x_4x4)
        x = self.conv6(x)
        
        x = x.view(x.size(0),-1)
        return x_4x4, x

class Motion_decoder(nn.Module):
    def __init__(self, deconv_channels: List[int]) -> None:
        super(Motion_decoder, self).__init__()

        self.deconv1 = self.deconv_layer(deconv_channels[0], deconv_channels[1])
        self.deconv2 = self.deconv_layer(deconv_channels[1], deconv_channels[2])
        if len(deconv_channels) == 4:
            self.deconv3 = self.deconv_layer(deconv_channels[2], deconv_channels[3])
        else:
            self.deconv3 = None
        self.deconvRecon = self.deconv_layer(deconv_channels[-1], 3)

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
        if self.deconv3 is not None:
            x = self.deconv3(x)
        
        if self.training:
            motion_image = self.deconvRecon(x)
            return x, motion_image
        else:
            return x

class Motion_attention(nn.Module):
    def __init__(self) -> None:
        super(Motion_attention, self).__init__()

        deconv_channels = [256, 128, 64] #assuming starting from 4x4 to 32x32
        self.motion_decoder = Motion_decoder(deconv_channels)
        self.conv = self.conv_layer(64, 128)

    def conv_layer(
        self,
        chIN,
        chOUT,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=True,
        pool_kernel=7
    ):
        conv = nn.Sequential(nn.Conv2d(chIN, chOUT, kernel_size, stride, padding, bias=bias),
                             nn.BatchNorm2d(chOUT),
                             nn.AvgPool2d(pool_kernel))
        return conv

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mi = None
        if self.training:
            motion_encoding, mi = self.motion_decoder(x)
        else:
            motion_encoding = self.motion_decoder(x)
        motion_encoding = self.conv(motion_encoding)
        motion_encoding = motion_encoding.view(motion_encoding.size(0),-1)
        return motion_encoding, mi

class Motion_DeeperAttention(nn.Module):
    def __init__(self) -> None:
        super(Motion_DeeperAttention, self).__init__()

        deconv_channels = [256, 128, 64] #assuming starting from 4x4 to 32x32
        self.motion_decoder = Motion_decoder(deconv_channels)
        self.conv1 = nn.Sequential(nn.Conv2d(64, 92, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.BatchNorm2d(92),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(92, 128, kernel_size=3, stride=1, padding=0, bias=True),
                                   nn.BatchNorm2d(128),
                                   nn.MaxPool2d(3, 2, 1),
                                   nn.AvgPool2d(7))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mi = None
        if self.training:
            motion_encoding, mi = self.motion_decoder(x)
        else:
            motion_encoding = self.motion_decoder(x)
        motion_encoding = self.conv1(motion_encoding)
        motion_encoding = self.conv2(motion_encoding)
        motion_encoding = motion_encoding.view(motion_encoding.size(0),-1)
        return motion_encoding, mi

class Motion_attention_64(nn.Module):
    def __init__(self) -> None:
        super(Motion_attention_64, self).__init__()

        deconv_channels = [256, 128, 64, 32] #assuming starting from 4x4 to 32x32
        self.motion_decoder = Motion_decoder(deconv_channels)
        self.conv1 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
                                   nn.BatchNorm2d(128),
                                   nn.AvgPool2d(7))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            motion_encoding, mi = self.motion_decoder(x)
        else:
            motion_encoding = self.motion_decoder(x)
            mi = None
        motion_encoding = self.conv1(motion_encoding)
        motion_encoding = self.conv2(motion_encoding)
        motion_encoding = motion_encoding.view(motion_encoding.size(0),-1)
        return motion_encoding, mi


class Attention(nn.Module):
    def __init__(self) -> None:
        super(Attention, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
                                  nn.BatchNorm2d(128),
                                  nn.AvgPool2d(7))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        motion_encoding = self.conv(motion_encoding)
        motion_encoding = motion_encoding.view(motion_encoding.size(0),-1)
        return motion_encoding
        