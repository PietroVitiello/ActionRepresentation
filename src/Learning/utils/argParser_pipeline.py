import torch
import torch.nn as nn
import torch.optim as optim

from ..BaselineCNN.models import BaselineCNN, Aux_BaselineCNN, LSTM_BaselineCNN, LSTM_largerBaseCNN
from ..AutoEncoder.models import SpatialAE_fc, StrengthSpatialAE_fc
from ..Stopping.models import Stopping_base

from ..training import Train

def model_choice(
    model_name,
    num_outputs,
    num_aux_outputs,
    recon_size,
    *args
) -> nn.Module:
    if model_name == "BaselineCNN":
        return BaselineCNN(*args)
    elif model_name == "Aux_BaselineCNN":
        return Aux_BaselineCNN(num_outputs, num_aux_outputs)
    elif model_name == "LSTM_BaselineCNN":
        return LSTM_BaselineCNN(*args)
    elif model_name == "LSTM_largerBaseCNN":
        return LSTM_largerBaseCNN(*args)
    elif model_name == "SpatialAE_fc":
        return SpatialAE_fc(num_outputs, num_aux_outputs, recon_size)
    elif model_name == "StrengthSpatialAE_fc":
        return StrengthSpatialAE_fc(num_outputs, num_aux_outputs, recon_size)
    elif model_name == "Stopping_base":
        return Stopping_base(num_outputs, num_aux_outputs)
    else:
        raise Exception("There is no such model available")

def train_model(train: Train, mode):
    if mode == 'eeVel':
        train.train_eeVel()
    elif mode == 'eeVel_aux':
        train.train_eeVelAux()
    elif mode == 'AE':
        train.train_AE(16)
    elif mode == 'stop':
        train.train_stopping()
    else:
        raise Exception("Training modality selected has not been recognized")