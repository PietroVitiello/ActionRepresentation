import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from .modules import SpatialSoftArgmax, SpatialSoftArgmax_strength
from .backbones import BaselineCNN_backbone, Motion_attention, Motion_DeeperAttention, Motion_attention_64, Motion_decoder, Attention

image_type = torch.Tensor

class MotionImage_attention(nn.Module):
    def __init__(self, num_outputs=6, num_aux_outputs=9) -> None:
        super(MotionImage_attention, self).__init__()

        self.cnn_backbone = BaselineCNN_backbone()
        self.mi_attention = Motion_attention()

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
        self.mi_attention.requires_grad_(False)
        self.reach_fc.requires_grad_(False)
        self.out.requires_grad_(False)
        self.aux.requires_grad_(False)

    def forward(self, x: image_type, train_stop: bool= None):
        feature_map4x4, x_conv = self.cnn_backbone(x)
        if self.training:
            if train_stop == False:
                mi_encoding, mi = self.mi_attention(feature_map4x4)
                x = self.reach_fc(x_conv)
                x = torch.mul(x, mi_encoding)
                x_aux = self.aux(x)
                x_out = self.out(x)
                return torch.cat((x_out, x_aux), dim=1), mi
            if train_stop == True:
                return self.stop(x_conv)
        else:
            mi_encoding, _ = self.mi_attention(feature_map4x4)
            reach_x = self.reach_fc(x_conv)
            reach_x = torch.mul(reach_x, mi_encoding)
            reach_x = self.out(reach_x)
            stop_signal = self.stop(x_conv)
            return (reach_x, stop_signal)

class MotionImage_DeeperAttention(nn.Module):
    def __init__(self, num_outputs=6, num_aux_outputs=9) -> None:
        super(MotionImage_DeeperAttention, self).__init__()

        self.cnn_backbone = BaselineCNN_backbone()
        self.mi_attention = Motion_DeeperAttention()

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
        self.mi_attention.requires_grad_(False)
        self.reach_fc.requires_grad_(False)
        self.out.requires_grad_(False)
        self.aux.requires_grad_(False)

    def forward(self, x: image_type, train_stop: bool= None):
        feature_map4x4, x_conv = self.cnn_backbone(x)
        if self.training:
            if train_stop == False:
                mi_encoding, mi = self.mi_attention(feature_map4x4)
                x = self.reach_fc(x_conv)
                x = torch.mul(x, mi_encoding)
                x_aux = self.aux(x)
                x_out = self.out(x)
                return torch.cat((x_out, x_aux), dim=1), mi
            if train_stop == True:
                return self.stop(x_conv)
        else:
            mi_encoding, _ = self.mi_attention(feature_map4x4)
            reach_x = self.reach_fc(x_conv)
            reach_x = torch.mul(reach_x, mi_encoding)
            reach_x = self.out(reach_x)
            stop_signal = self.stop(x_conv)
            return (reach_x, stop_signal)

class MotionImage_attention_64(nn.Module):
    def __init__(self, num_outputs=6, num_aux_outputs=9) -> None:
        super(MotionImage_attention_64, self).__init__()

        self.cnn_backbone = BaselineCNN_backbone()
        self.mi_attention = Motion_attention_64()

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
        self.mi_attention.requires_grad_(False)
        self.reach_fc.requires_grad_(False)
        self.out.requires_grad_(False)
        self.aux.requires_grad_(False)

    def forward(self, x: image_type, train_stop: bool= None):
        feature_map4x4, x_conv = self.cnn_backbone(x)
        if self.training:
            if train_stop == False:
                mi_encoding, mi = self.mi_attention(feature_map4x4)
                x = self.reach_fc(x_conv)
                x = torch.mul(x, mi_encoding)
                x_aux = self.aux(x)
                x_out = self.out(x)
                return torch.cat((x_out, x_aux), dim=1), mi
            if train_stop == True:
                return self.stop(x_conv)
        else:
            mi_encoding, _ = self.mi_attention(feature_map4x4)
            reach_x = self.reach_fc(x_conv)
            reach_x = torch.mul(reach_x, mi_encoding)
            reach_x = self.out(reach_x)
            stop_signal = self.stop(x_conv)
            return (reach_x, stop_signal)

class MotionImage_auxiliary(nn.Module):
    def __init__(self, num_outputs=6, num_aux_outputs=9) -> None:
        super(MotionImage_auxiliary, self).__init__()

        self.cnn_backbone = BaselineCNN_backbone()
        deconv_channels = [256, 128, 64] #assuming starting from 4x4 to 32x32
        self.motion_decoder = Motion_decoder(deconv_channels)

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
        self.motion_decoder.requires_grad_(False)
        self.reach_fc.requires_grad_(False)
        self.out.requires_grad_(False)
        self.aux.requires_grad_(False)

    def forward(self, x: image_type, train_stop: bool= None):
        feature_map4x4, x_conv = self.cnn_backbone(x)
        if self.training:
            if train_stop == False:
                _, mi = self.motion_decoder(feature_map4x4)
                x = self.reach_fc(x_conv)
                x_aux = self.aux(x)
                x_out = self.out(x)
                return torch.cat((x_out, x_aux), dim=1), mi
            if train_stop == True:
                return self.stop(x_conv)
        else:
            reach_x = self.reach_fc(x_conv)
            reach_x = self.out(reach_x)
            stop_signal = self.stop(x_conv)
            return (reach_x, stop_signal)

class MI_Net_indepAE(nn.Module):
    def __init__(self, num_outputs=6, num_aux_outputs=9) -> None:
        super(MI_Net_indepAE, self).__init__()

        self.cnn_backbone = BaselineCNN_backbone()
        deconv_channels = [256, 128, 64] #assuming starting from 4x4 to 32x32
        self.motion_decoder = Motion_decoder(deconv_channels)
        self.mi_attention = Attention()

        self.reach_fc = nn.Sequential(nn.Linear(256, 128),
                                      nn.ReLU())

        self.aux = nn.Linear(128, num_aux_outputs)
        self.out = nn.Linear(128, num_outputs)

        self.stop = nn.Sequential(nn.Linear(256, 64),
                                  nn.ReLU(),
                                  nn.Linear(64, 1),
                                  nn.Sigmoid())

        self.freeze_reach()

    def freeze_reach(self):
        self.cnn_backbone.requires_grad_(True)
        self.motion_decoder.requires_grad_(True)

        self.mi_attention.requires_grad_(False)
        self.reach_fc.requires_grad_(False)
        self.out.requires_grad_(False)
        self.aux.requires_grad_(False)
        self.stop.requires_grad_(False)

    def unfreeze_reach(self):
        self.cnn_backbone.requires_grad_(False)
        self.motion_decoder.requires_grad_(False)

        self.mi_attention.requires_grad_(True)
        self.reach_fc.requires_grad_(True)
        self.out.requires_grad_(True)
        self.aux.requires_grad_(True)
        self.stop.requires_grad_(False)

    def freeze_backbone(self):
        self.cnn_backbone.requires_grad_(False)
        self.motion_decoder.requires_grad_(False)
        self.mi_attention.requires_grad_(False)
        self.reach_fc.requires_grad_(False)
        self.out.requires_grad_(False)
        self.aux.requires_grad_(False)
        self.stop.requires_grad_(True)

    def forward(self, x: image_type, train_actions: bool= True, train_stop: bool= None):
        feature_map4x4, x_conv = self.cnn_backbone(x)
        if self.training:
            if train_actions == False:
                _, mi = self.motion_decoder(feature_map4x4)
                return mi
            elif train_stop == False:
                mi_encoding, mi = self.motion_decoder(feature_map4x4)
                mi_encoding = self.mi_attention(mi_encoding)
                x = self.reach_fc(x_conv)
                x = torch.mul(x, mi_encoding)
                x_aux = self.aux(x)
                x_out = self.out(x)
                return torch.cat((x_out, x_aux), dim=1), mi
            elif train_stop == True:
                return self.stop(x_conv)
        else:
            mi_encoding = self.motion_decoder(feature_map4x4)
            mi_encoding = self.mi_attention(mi_encoding)
            reach_x = self.reach_fc(x_conv)
            reach_x = torch.mul(reach_x, mi_encoding)
            reach_x = self.out(reach_x)
            stop_signal = self.stop(x_conv)
            return (reach_x, stop_signal)

