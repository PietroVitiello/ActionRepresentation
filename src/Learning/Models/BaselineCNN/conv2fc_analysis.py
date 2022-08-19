import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from .nn_modules import *

class ReduceTo1x1(nn.Module):
    def __init__(self, num_outputs=6, num_aux_outputs=9):
        super(ReduceTo1x1,self).__init__()

        self.encoder = CNN_encoder()
        self.conv4 = conv_layer(192, 256)
        self.conv5 = conv_layer(256, 256)

        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0, bias=True),
                                   nn.ReLU(inplace=True))

        self.action_predictor = FC_predictor(num_outputs, num_aux_outputs)

    def forward(self, x):

        x = self.encoder(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        
        x = x.view(x.size(0),-1)
        x = self.action_predictor(x)
        return x

class AveragePool(nn.Module):
    def __init__(self, num_outputs=6, num_aux_outputs=9):
        super(AveragePool,self).__init__()

        self.encoder = CNN_encoder()
        self.global_pooling = nn.AvgPool2d(kernel_size=8)
        self.action_predictor = FC_predictor(num_outputs, num_aux_outputs)

    def forward(self, x):

        x = self.encoder(x)
        x = self.global_pooling(x)
        
        x = x.view(x.size(0),-1)
        x = self.action_predictor(x)
        return x

class MaximumPool(nn.Module):
    def __init__(self, num_outputs=6, num_aux_outputs=9):
        super(MaximumPool,self).__init__()

        self.encoder = CNN_encoder()
        self.global_pooling = nn.MaxPool2d(kernel_size=8)
        self.action_predictor = FC_predictor(num_outputs, num_aux_outputs)

    def forward(self, x):

        x = self.encoder(x)
        x = self.global_pooling(x)
        
        x = x.view(x.size(0),-1)
        x = self.action_predictor(x)
        return x

class Flattening(nn.Module):
    def __init__(self, num_outputs=6, num_aux_outputs=9):
        super(Flattening,self).__init__()

        self.encoder = CNN_encoder()
        self.conv4 = conv_layer(192, 64)

        self.action_predictor = FC_predictor(num_outputs, num_aux_outputs)

    def forward(self, x):

        x = self.encoder(x)
        x = self.conv4(x)
        
        x = x.view(x.size(0),-1)
        x = self.action_predictor(x)
        return x

class CoordReduceTo1x1(nn.Module):
    def __init__(self, num_outputs=6, num_aux_outputs=9):
        super(CoordReduceTo1x1,self).__init__()

        self.conv1 = CoordConv_block(3, 64)
        self.conv2 = CoordConv_block(64, 128)
        self.conv3 = CoordConv_block(128, 192)
        self.conv4 = CoordConv_block(192, 256)
        self.conv5 = CoordConv_block(256, 256)
        self.conv6 = nn.Sequential(CoordConv(256, 256, kernel_size=2, stride=1, padding=0, bias=True),
                                   nn.ReLU(inplace=True))

        self.action_predictor = FC_predictor(num_outputs, num_aux_outputs)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        
        x = x.view(x.size(0),-1)
        x = self.action_predictor(x)
        return x

class CoordAveragePool(nn.Module):
    def __init__(self, num_outputs=6, num_aux_outputs=9):
        super(CoordAveragePool,self).__init__()

        self.conv1 = CoordConv_block(3, 64)
        self.conv2 = CoordConv_block(64, 128)
        self.conv3 = CoordConv_block(128, 192)
        self.global_pooling = nn.AvgPool2d(kernel_size=8)

        self.action_predictor = FC_predictor(num_outputs, num_aux_outputs)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pooling(x)
        
        x = x.view(x.size(0),-1)
        x = self.action_predictor(x)
        return x
