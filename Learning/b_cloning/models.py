import torch
from torch.nn import Conv2d, MaxPool2d
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

class ResNet(nn.Module):
    def __init__(self,num_outputs = 13,fconv=[3,1,1]):
        super(ResNet,self).__init__()

        self.conv1 = nn.Sequential(Conv2d(3,64,kernel_size= fconv[0],stride=fconv[1],padding=fconv[2],bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(Conv2d(64,128,kernel_size=fconv[0],stride=fconv[1],padding=fconv[2],bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(Conv2d(128, 256, kernel_size=fconv[0], stride=fconv[1], padding=fconv[2], bias=False),
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