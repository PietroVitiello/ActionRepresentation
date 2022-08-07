import torch
import torch.nn as nn
from .backbones import BaselineCNN_backbone

class Stopping_base(nn.Module):
    def __init__(self, num_outputs=6, num_aux_outputs=9):
        super(Stopping_base, self).__init__()

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
        self.stop = nn.Sequential(nn.Linear(128, 1),
                                  nn.Sigmoid()) # stopping bunary neuron 

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
            x_stop = self.stop(x)
            x = torch.cat((x_out, x_aux, x_stop), dim=1)
        else:
            x_out = self.out(x)
            x_stop = self.stop(x)
            x = (x_out, x_stop)
        return x

class Stop_AuxBaselineCNN(nn.Module):
    def __init__(self, num_outputs=6, num_aux_outputs=9):
        super(Stop_AuxBaselineCNN, self).__init__()

        self.cnn_backbone = BaselineCNN_backbone()

        self.reach_fc = nn.Sequential(nn.Linear(256, 128),
                                      nn.ReLU())

        self.aux = nn.Linear(128, num_aux_outputs)
        self.out = nn.Linear(128, num_outputs)

        self.stop = nn.Sequential(nn.Linear(256, 64),
                                  nn.ReLU(),
                                  nn.Linear(64, 1),
                                  nn.Sigmoid())

    def freeze_backbone(self):
        self.cnn_backbone.requires_grad_(False)
        self.reach_fc.requires_grad_(False)
        self.out.requires_grad_(False)
        self.aux.requires_grad_(False)

    def forward(self, x, train_stop: bool=None):
        if self.training:
            if train_stop == False:
                x = self.cnn_backbone(x)
                x = self.reach_fc(x)
                x_aux = self.aux(x)
                x_out = self.out(x)
                return torch.cat((x_out, x_aux), dim=1)
            if train_stop == True:
                # with torch.no_grad():
                x: torch.Tensor = self.cnn_backbone(x)
                # x.requires_grad = True
                return self.stop(x)
        else:
            x = self.cnn_backbone(x)

            reach_x = self.reach_fc(x)
            reach_x = self.out(reach_x)
            stop_signal = self.stop(x)
            return (reach_x, stop_signal)
