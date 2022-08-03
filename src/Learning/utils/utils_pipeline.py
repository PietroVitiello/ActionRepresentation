import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

from ..Models.BaselineCNN.models import BaselineCNN, Aux_BaselineCNN, LSTM_BaselineCNN, LSTM_largerBaseCNN
from ..Models.AutoEncoder.models import SpatialAE_fc, StrengthSpatialAE_fc
from ..Models.Stopping.models import Stopping_base, Stop_AuxBaselineCNN
from ..Models.MotionIMG.models import MotionImage_attention
from ..Models.PureAE.models import Pure_SimpleAE, Pure_SimpleAE_mediumDec,Pure_SimpleAE_vlargeDec

from ..training import Train
from ..testing import Test

###################### Training ######################

def model_choice(
    model_name,
    num_outputs,
    num_aux_outputs,
    recon_size
) -> nn.Module:
    if model_name == "BaselineCNN":
        return BaselineCNN(num_outputs)
    elif model_name == "Aux_BaselineCNN":
        return Aux_BaselineCNN(num_outputs, num_aux_outputs)
    elif model_name == "LSTM_BaselineCNN":
        return LSTM_BaselineCNN(num_outputs, num_aux_outputs)
    elif model_name == "LSTM_largerBaseCNN":
        return LSTM_largerBaseCNN(num_outputs, num_aux_outputs)
    elif model_name == "SpatialAE_fc":
        return SpatialAE_fc(num_outputs, num_aux_outputs, recon_size)
    elif model_name == "StrengthSpatialAE_fc":
        return StrengthSpatialAE_fc(num_outputs, num_aux_outputs, recon_size)
    elif model_name == "Stopping_base":
        return Stopping_base(num_outputs, num_aux_outputs)
    elif model_name == "Stop_AuxBaselineCNN":
        return Stop_AuxBaselineCNN(num_outputs, num_aux_outputs)
    elif model_name == "MotionImage_attention":
        return MotionImage_attention(num_outputs, num_aux_outputs)

    elif model_name == "Pure_SimpleAE":
        return Pure_SimpleAE()
    elif model_name == "Pure_SimpleAE_mediumDec":
        return Pure_SimpleAE_mediumDec()
    elif model_name == "Pure_SimpleAE_vlargeDec":
        return Pure_SimpleAE_vlargeDec()
    else:
        raise Exception("There is no such model available")

def train_model(train: Train, mode):
    if mode == 'eeVel':
        train.train_eeVel()
    elif mode == 'eeVel_aux':
        train.train_eeVelAux()
    elif mode == 'AE':
        train.train_AE()
    elif mode == 'stop':
        train.train_stopping()
    elif mode == 'aux_stopIndividual':
        train.train_auxStopIndividual()
    elif mode == 'motion_image':
        train.train_MotionImage()

    elif mode == 'pureAE':
        train.train_PureAE()
    else:
        raise Exception("Training modality selected has not been recognized")

def uselessParams(mode: str):
    useless_keys = []
    # if model == "BaselineCNN":
    #     useless_keys.append("stopping_loss")
    #     useless_keys.append("reconstruction_size")
    #     useless_keys.append("num_aux_outputs")
    # elif model == "Aux_BaselineCNN":
    #     useless_keys.append("stopping_loss")
    #     useless_keys.append("reconstruction_size")
    # elif model == "LSTM_BaselineCNN":
    #     useless_keys.append("stopping_loss")
    #     useless_keys.append("reconstruction_size")
    # elif model == "LSTM_largerBaseCNN":
    #     useless_keys.append("stopping_loss")
    #     useless_keys.append("reconstruction_size")
    # elif model == "SpatialAE_fc":
    #     useless_keys.append("stopping_loss")
    # elif model == "StrengthSpatialAE_fc":
    #     useless_keys.append("stopping_loss")
    # elif model == "Stopping_base":
    #     useless_keys.append("reconstruction_size")
    # else:
    #     raise Exception("There is no such model available")

    if mode == 'eeVel':
        useless_keys.append("stopping_loss")
        useless_keys.append("reconstruction_size")
        useless_keys.append("num_aux_outputs")
    elif mode == 'eeVel_aux':
        useless_keys.append("stopping_loss")
        useless_keys.append("reconstruction_size")
    elif mode == 'AE':
        useless_keys.append("stopping_loss")
    elif mode == 'stop':
        useless_keys.append("reconstruction_size")
    elif mode == 'aux_stopIndividual':
        useless_keys.append("reconstruction_size")
    elif mode == 'motion_image':
        useless_keys.append("reconstruction_size")

    elif mode == 'pureAE':
        useless_keys.append("stopping_loss")
        useless_keys.append("reconstruction_size")
        useless_keys.append("num_outputs")
        useless_keys.append("num_aux_outputs")
    else:
        raise Exception("Training modality selected has not been recognized")
    return useless_keys

###################### Testing ######################

def getModelData(model_filename: str):
    with open("Learning/TrainedModels/model_config.yaml", 'r') as file:
        configs = yaml.safe_load(file)
    model_data = configs[model_filename]
    model_name = model_data["model_name"]
    dataset_name = model_data["data_folder"]
    num_outputs = model_data["num_outputs"]
    num_aux_outputs = model_data["num_aux_outputs"]
    try:
        recon_size = model_data["reconstruction_size"]
    except KeyError:
        recon_size = None
    constrained = False if num_outputs<6 else True
    return model_name, constrained, dataset_name, (num_outputs, num_aux_outputs, recon_size)

def getRestriction(restriction: str, dataset_name: str):
    if restriction == "same":
        with open("Demos/Dataset/descriptions.yaml", 'r') as file:
            configs = yaml.safe_load(file)
        dataset_data = configs[dataset_name]
        restriction = dataset_data["boundary_restriction"]
    return restriction

def testMethod(test: Test, model_name: str, constrained: bool):
    LSTM_models = ["LSTM_largerBaseCNN", "LSTM_BaselineCNN"]
    stopping_models = ["Stopping_base", "Stop_AuxBaselineCNN", "MotionImage_attention"]
    if model_name in LSTM_models:
        return test.test_eeVel_LSTM()
    elif model_name in stopping_models:
        return test.test_eeVelGrasp(constrained)
    else:
        return test.test_eeVel(constrained)