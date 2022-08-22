import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class conv_layer(nn.Module):
    def __init__(
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
        super(conv_layer, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(chIN, chOUT, kernel_size, stride, padding, bias=bias),
                            nn.BatchNorm2d(chOUT),
                            nn.MaxPool2d(pool_kernel, pool_stride, pool_padding),
                            nn.ReLU(inplace=True))
        
    def forward(self, x):
        return self.conv(x)

class CNN_encoder(nn.Module):
    def __init__(self, channels=[64, 128, 192]):
        '''
        Simple convolutional encoder from 64x64 to 8x8
        '''
        super(CNN_encoder, self).__init__()

        self.conv1 = conv_layer(3, channels[0])
        self.conv2 = conv_layer(channels[0], channels[1])
        self.conv3 = conv_layer(channels[1], channels[2])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class FC_predictor(nn.Module):
    def __init__(self, num_outputs=6, num_aux_outputs=9):
        super(FC_predictor, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(256, 128),
                                 nn.ReLU())

        self.aux = nn.Linear(128, num_aux_outputs)
        self.out = nn.Linear(128, num_outputs)

    def forward(self, x):
        x = self.fc1(x)
        if self.training:
            x_aux = self.aux(x)
            x_out = self.out(x)
            x = torch.cat((x_out, x_aux), dim=1)
        else:
            x = self.out(x)
        return x

class CoordConv(nn.Module):
    def __init__(
        self,
        chIN,
        chOUT,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
    ) -> None:

        super(CoordConv, self).__init__()
        self.conv = nn.Conv2d(chIN+2, chOUT, kernel_size, stride, padding, bias=bias)

    def addCoord(self, batch: torch.Tensor):
        device = batch.device
        batch_size, x, y = batch.shape[0], batch.shape[2], batch.shape[3]

        xx_ones = torch.ones([batch_size, x], dtype=torch.int32)   # e.g. (batch, 64)
        xx_ones = torch.unsqueeze(xx_ones, -1)                     # e.g. (batch, 64, 1)
        xx_range = torch.tile(torch.unsqueeze(torch.arange(0, y), 0), 
                            [batch_size, 1])                       # e.g. (batch, 64)
        xx_range = torch.unsqueeze(xx_range, 1).to(dtype=torch.int32)                    # e.g. (batch, 1, 64)
        xx_channel = torch.matmul(xx_ones, xx_range)               # e.g. (batch, 64, 64)
        xx_channel = torch.unsqueeze(xx_channel, 1)               # e.g. (batch, 1, 64, 64)


        yy_ones = torch.ones([batch_size, y], dtype=torch.int32)   # e.g. (batch, 64)
        yy_ones = torch.unsqueeze(yy_ones, 1)                      # e.g. (batch, 1, 64)
        yy_range = torch.tile(torch.unsqueeze(torch.arange(0, x), 0),
                            [batch_size, 1])                       # (batch, 64)
        yy_range = torch.unsqueeze(yy_range, -1).to(dtype=torch.int32)                   # e.g. (batch, 64, 1)
        yy_channel = torch.matmul(yy_range, yy_ones)               # e.g. (batch, 64, 64)
        yy_channel = torch.unsqueeze(yy_channel, 1)               # e.g. (batch, 1, 64, 64)


        xx_channel = xx_channel.to(device=device, dtype=torch.float32) / (x - 1)
        yy_channel = yy_channel.to(device=device, dtype=torch.float32) / (y - 1)
        xx_channel = xx_channel*2 - 1                           # [-1,1]
        yy_channel = yy_channel*2 - 1
        return torch.concat((batch, xx_channel, yy_channel), axis=1)    # e.g. (batch, 64, 64, c+2)

    def forward(self, x: torch.Tensor):
        x = self.addCoord(x)
        x = self.conv(x)
        return x

class CoordConv_block(nn.Module):
    def __init__(
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
    ) -> None:

        super(CoordConv_block, self).__init__()

        self.conv = nn.Sequential(CoordConv(chIN, chOUT, kernel_size, stride, padding, bias=bias),
                                  nn.BatchNorm2d(chOUT),
                                  nn.MaxPool2d(pool_kernel, pool_stride, pool_padding),
                                  nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        return x

