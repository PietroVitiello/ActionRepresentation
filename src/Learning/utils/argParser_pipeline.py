import torch
import torch.nn as nn
import torch.optim as optim

from ..BaselineCNN.models import BaselineCNN, Aux_BaselineCNN, LSTM_BaselineCNN, LSTM_largerBaseCNN
from ..AutoEncoder.models import SpatialAE_fc

from ..training import Train

def model_choice(model_name, *args) -> nn.Module:
    if model_name == "BaselineCNN":
        return BaselineCNN(*args)
    elif model_name == "Aux_BaselineCNN":
        return Aux_BaselineCNN(*args)
    elif model_name == "LSTM_BaselineCNN":
        return LSTM_BaselineCNN(*args)
    elif model_name == "LSTM_largerBaseCNN":
        return LSTM_largerBaseCNN(*args)
    elif model_name == "SpatialAE_fc":
        return SpatialAE_fc(*args)
    else:
        raise Exception("There is no such model available")

def train_model(train: Train, mode):
    if mode == 'eeVel':
        train.train_eeVel()
    elif mode == 'eeVel_aux':
        train.train_eeVelAux()
    elif mode == 'AE':
        train.train_AE(16)
    else:
        raise Exception("Training modality selected has not been recognized")