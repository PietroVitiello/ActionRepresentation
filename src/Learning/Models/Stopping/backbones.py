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

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        
        x = x.view(x.size(0),-1)
        return x
