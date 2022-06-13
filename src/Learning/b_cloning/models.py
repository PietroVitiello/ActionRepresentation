import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import sampler
from torchvision import datasets, transforms
from os.path import dirname, join, abspath
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import PIL
import PIL.Image as Image
import random

class BaselineCNN(nn.Module):
    def __init__(self, num_outputs=6):
        super(BaselineCNN,self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128,kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(192),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0, bias=True),
                                   nn.ReLU(inplace=True))

        self.fc = nn.Sequential(nn.Linear(256, 128),
                                nn.ReLU(),
                                nn.Linear(128, num_outputs))

    def forward(self,x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        
        x = x.view(x.size(0),-1)
        x = self.fc(x)

        return x

class SimpleCNN(nn.Module):
    def __init__(self, num_outputs=13, fconv=[3,1,1]):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3,64,kernel_size= fconv[0],stride=fconv[1],padding=fconv[2],bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64,128,kernel_size=fconv[0],stride=fconv[1],padding=fconv[2],bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=fconv[0], stride=fconv[1], padding=fconv[2], bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(inplace=True))
        self.fc = nn.Linear(256*8*8,num_outputs)

    def forward(self,x):

        x = self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)

        return x


class Aux_BaselineCNN(nn.Module):
    def __init__(self, num_aux_outputs, num_outputs=6):
        super(Aux_BaselineCNN, self).__init__()

        self.conv1 = self.conv_layer(3, 64)
        self.conv2 = self.conv_layer(64, 128)
        self.conv3 = self.conv_layer(128, 192)
        self.conv4 = self.conv_layer(192, 256)
        self.conv5 = self.conv_layer(256, 256)

        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0, bias=True),
                                   nn.ReLU(inplace=True))

        self.fc1 = nn.Sequential(nn.Linear(256, 128),
                                 nn.ReLU())

        self.aux = nn.Linear(128, num_aux_outputs)
        self.out = nn.Linear(128, num_outputs)

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
        x = self.fc1(x)

        if self.training:
            x_aux = self.aux(x)
            x_out = self.out(x)
            x = torch.cat((x_out, x_aux), dim=1)
        else:
            x = self.out(x)
        return x


class LSTM_BaselineCNN(nn.Module):
    def __init__(self, num_aux_outputs, num_outputs=6):
        super(LSTM_BaselineCNN, self).__init__()

        self.conv1 = self.conv_layer(3, 64)
        self.conv2 = self.conv_layer(64, 128)
        self.conv3 = self.conv_layer(128, 192)
        self.conv4 = self.conv_layer(192, 256)
        self.conv5 = self.conv_layer(256, 256)

        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0, bias=True),
                                   nn.ReLU(inplace=True))

        self.lstm = nn.LSTM(
                input_size = 256,
                hidden_size= 128,
                num_layers= 1,
                batch_first=True
            )

        self.aux = nn.Linear(128, num_aux_outputs)
        self.out = nn.Linear(128, num_outputs)

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

    def start_newSeq(self):
        self.h = torch.zeros((1, 128))
        self.c = torch.zeros((1, 128))

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)    

        if self.training:
            x = x.view(1, x.size(0),-1)
            x, _ = self.lstm(x)
            x = x.squeeze()
            x_aux = self.aux(x)
            x_out = self.out(x)
            x = torch.cat((x_out, x_aux), dim=1)
        else:
            x = x.view(x.size(0),-1)
            x, (self.h, self.c) = self.lstm(x, (self.h, self.c))
            x = x.squeeze()
            x = self.out(x)
        return x


class LSTM_largerBaseCNN(nn.Module):
    def __init__(self, num_aux_outputs, num_outputs=6):
        super(LSTM_largerBaseCNN, self).__init__()

        self.conv1 = self.conv_layer(3, 64)
        self.conv2 = self.conv_layer(64, 128)
        self.conv3 = self.conv_layer(128, 192)
        self.conv4 = self.conv_layer(192, 256)
        self.conv5 = self.conv_layer(256, 256)

        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0, bias=True),
                                   nn.ReLU(inplace=True))

        self.lstm = nn.LSTM(
                input_size = 256,
                hidden_size= 128,
                num_layers= 1,
                batch_first=True
            )

        self.fc1 = nn.Sequential(nn.Linear(128, 64),
                                 nn.ReLU())

        self.aux = nn.Linear(64, num_aux_outputs)
        self.out = nn.Linear(64, num_outputs)

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

    def start_newSeq(self):
        self.h = torch.zeros((1, 128))
        self.c = torch.zeros((1, 128))

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)    

        if self.training:
            x = x.view(1, x.size(0),-1)
            x, _ = self.lstm(x)
            x = x.squeeze()
            x = self.fc1(x)
            x_aux = self.aux(x)
            x_out = self.out(x)
            x = torch.cat((x_out, x_aux), dim=1)
        else:
            x = x.view(x.size(0),-1)
            x, (self.h, self.c) = self.lstm(x, (self.h, self.c))
            x = x.squeeze()
            x = self.fc1(x)
            x = self.out(x)
        return x