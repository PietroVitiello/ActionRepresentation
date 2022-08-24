import yaml
import numpy as np
import pandas as pd

from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from ..Models.BaselineCNN.models import BaselineCNN, Aux_BaselineCNN, LSTM_BaselineCNN, LSTM_largerBaseCNN
from ..Models.AutoEncoder.models import SpatialAE_fc, StrengthSpatialAE_fc
from ..Models.Stopping.models import Stopping_base, Stop_AuxBaselineCNN
from ..Models.MotionIMG.models import MotionImage_attention
from ..Models.PureAE.models import Pure_SimpleAE, Pure_SimpleAE_mediumDec,Pure_SimpleAE_vlargeDec
from ..Models.BaselineCNN.conv2fc_analysis import ReduceTo1x1, AveragePool, MaximumPool, Flattening, CoordReduceTo1x1, CoordAveragePool

# from ..training import Train
from ..Testing.testing import Test
from ..Training.train_eeVel import Train_eeVel
from ..Training.train_eeVelAux import Train_eeVelAux
from ..Training.train_Aux_wnb import Train_eeVelAux_wandb
from ..Training.train_AuxStop import Train_AuxStop
from ..Training.train_AE import Train_AE
from ..Training.train_wnb import Train_AE_wandb
from ..Training.train import Training

from ..TrainLoaders.trainloader import SimDataset
from ..TrainLoaders.TL_eeVel import TL_eeVel
from ..TrainLoaders.TL_aux import TL_aux
from ..TrainLoaders.TL_stop import TL_stop
from ..TrainLoaders.TL_onlyStop import TL_onlyStop
from ..TrainLoaders.TL_MI import TL_motionImage

###################### Data ######################

class dataset_pipeline():

    def __init__(
        self,
        dataset_path,
        train_val_split: float,
        n_demos: int=None
    ) -> None:
        self.dataset_path = dataset_path
        df = pd.read_csv(dataset_path + "data.csv")
        self.train_ids, self.val_ids = SimDataset.get_train_val_ids(df, train_val_split, n_demos)

    def get_dataset_with_val( 
        self,
        transform = None,
        dataset_mode: str="eeVel",
        filter_stop: bool=False
    ) -> Tuple[SimDataset]:

        train_dataset = self.get_dataset(
            self.dataset_path,
            transform,
            dataset_mode,
            filter_stop,
            self.train_ids
        )
        val_dataset = self.get_dataset(
            self.dataset_path,
            transform,
            dataset_mode,
            filter_stop,
            self.val_ids
        )
        return train_dataset, val_dataset

    @staticmethod
    def get_dataset( 
        dataset_path,
        transform = None,
        dataset_mode: str="eeVel",
        filter_stop: bool=False,
        considered_indices: np.ndarray = None
    ) -> SimDataset:

        if dataset_mode == "eeVel":
            return TL_eeVel(
                dataset_path,
                transform,
                filter_stop,
                considered_indices
            )
        elif dataset_mode == "aux":
            return TL_aux(
                dataset_path,
                transform,
                filter_stop,
                considered_indices=considered_indices
            )
        elif dataset_mode == "stop":
            return TL_stop(
                dataset_path,
                transform,
                filter_stop,
                considered_indices=considered_indices
            )
        elif dataset_mode == "onlyStop":
            return TL_onlyStop(
                dataset_path,
                transform,
                filter_stop,
                considered_indices=considered_indices
            )
        elif dataset_mode == "motionImage":
            return TL_motionImage(
                dataset_path,
                transform,
                filter_stop,
                delta_steps=5,
                considered_indices=considered_indices
            )
        elif dataset_mode == "none":
            return None
        else:
            raise Exception("The selected dataset mode is not supported")

###################### Training ######################

def get_trainer(
        training_type: str,
        model: torch.nn.Module,
        model_name: str,
        dataset: DataLoader,
        val_datasets: Tuple[DataLoader],
        stopping_dataset: DataLoader,
        transform: T,
        use_gpu: bool,
        epochs: int,
        stopping_epochs: int,
        batch_size: int,
        optimiser: str,
        lr: float,
        weight_decay: float = 1e-7,
        loss: str = 'MSE',
        stopping_loss: str = 'BCE',
        recon_size: int = 16
    ) -> Training:
    
        if training_type == 'eeVel':
            return Train_eeVel(
                        model,
                        dataset,
                        use_gpu,
                        epochs,
                        batch_size,
                        optimiser,
                        lr,
                        weight_decay,
                        loss
                    )
        elif training_type == 'eeVel_aux':
            return Train_eeVelAux(
                        model,
                        dataset,
                        use_gpu,
                        epochs,
                        batch_size,
                        optimiser,
                        lr,
                        weight_decay,
                        loss
                    )
        elif training_type == 'eeVel_aux_wandb':
            return Train_eeVelAux_wandb(
                        model,
                        model_name,
                        dataset,
                        val_datasets,
                        transform,
                        use_gpu,
                        epochs,
                        batch_size,
                        optimiser,
                        lr,
                        weight_decay,
                        loss
                    )
        elif training_type == 'aux_stop_wandb':
            return Train_AuxStop(
                        model,
                        model_name,
                        dataset,
                        val_datasets,
                        stopping_dataset,
                        transform,
                        use_gpu,
                        epochs,
                        stopping_epochs,
                        batch_size,
                        optimiser,
                        lr,
                        weight_decay,
                        loss,
                        stopping_loss,
                        recon_size
                    )
        elif training_type == 'AE':
            return Train_AE(
                        model,
                        dataset,
                        val_datasets,
                        stopping_dataset,
                        transform,
                        use_gpu,
                        epochs,
                        stopping_epochs,
                        batch_size,
                        optimiser,
                        lr,
                        weight_decay,
                        loss,
                        stopping_loss,
                        recon_size
                    )
        elif training_type == 'AE_wandb':
            return Train_AE_wandb(
                        model,
                        model_name,
                        dataset,
                        val_datasets,
                        stopping_dataset,
                        transform,
                        use_gpu,
                        epochs,
                        stopping_epochs,
                        batch_size,
                        optimiser,
                        lr,
                        weight_decay,
                        loss,
                        stopping_loss,
                        recon_size
                    )
        # elif training_type == 'stop':
        #     train.train_stopping()
        # elif training_type == 'aux_stopIndividual':
        #     train.train_auxStopIndividual()
        # elif training_type == 'motion_image':
        #     train.train_MotionImage()

        # elif training_type == 'pureAE':
        #     train.train_PureAE()
        else:
            raise Exception("Training modality selected has not been recognized")

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

    elif model_name == "ReduceTo1x1":
        return ReduceTo1x1(num_outputs, num_aux_outputs)
    elif model_name == "AveragePool":
        return AveragePool(num_outputs, num_aux_outputs)
    elif model_name == "MaximumPool":
        return MaximumPool(num_outputs, num_aux_outputs)
    elif model_name == "Flattening":
        return Flattening(num_outputs, num_aux_outputs)
    elif model_name == "CoordReduceTo1x1":
        return CoordReduceTo1x1(num_outputs, num_aux_outputs)
    elif model_name == "CoordAveragePool":
        return CoordAveragePool(num_outputs, num_aux_outputs)

    elif model_name == "Pure_SimpleAE":
        return Pure_SimpleAE()
    elif model_name == "Pure_SimpleAE_mediumDec":
        return Pure_SimpleAE_mediumDec()
    elif model_name == "Pure_SimpleAE_vlargeDec":
        return Pure_SimpleAE_vlargeDec()
    else:
        raise Exception("There is no such model available")

# def train_model(train: Train, mode):
#     if mode == 'eeVel':
#         train.train_eeVel()
#     elif mode == 'eeVel_aux':
#         train.train_eeVelAux()
#     elif mode == 'AE':
#         train.train_AE()
#     elif mode == 'stop':
#         train.train_stopping()
#     elif mode == 'aux_stopIndividual':
#         train.train_auxStopIndividual()
#     elif mode == 'motion_image':
#         train.train_MotionImage()

#     elif mode == 'pureAE':
#         train.train_PureAE()
#     else:
#         raise Exception("Training modality selected has not been recognized")

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
    elif mode == 'eeVel_aux_wandb':
        useless_keys.append("stopping_loss")
        useless_keys.append("reconstruction_size")
    elif mode == 'AE':
        useless_keys.append("stopping_loss")
    elif mode == 'stop':
        useless_keys.append("reconstruction_size")
    elif mode == 'aux_stopIndividual':
        useless_keys.append("reconstruction_size")
    elif mode == 'aux_stop_wandb':
        useless_keys.append("reconstruction_size")
    elif mode == 'motion_image':
        useless_keys.append("reconstruction_size")
    elif mode == 'AE_wandb':
        useless_keys.append("reconstruction_size")

    elif mode == 'pureAE':
        useless_keys.append("stopping_loss")
        useless_keys.append("reconstruction_size")
        useless_keys.append("num_outputs")
        useless_keys.append("num_aux_outputs")
    else:
        raise Exception("Training modality selected has not been recognized")
    return useless_keys

def return_data_stats(stats):
    input_metrics = [stats[0][1], stats[0][0]]
    recon_metrics = [stats[1][1], stats[1][0]]
    eeVel_metrics = [stats[2][1][:6], stats[2][0][:6]]
    aux_metrics = [stats[2][1][6:], stats[2][0][6:]]
    return input_metrics, recon_metrics, eeVel_metrics, aux_metrics

###################### Testing ######################

def getModelData(model_filename: str, config_filename: str):
    with open(f"Learning/TrainedModels/{config_filename}.yaml", 'r') as file:
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

def testMethod(test: Test, model_name: str, constrained: bool, use_saved_locations: bool):
    LSTM_models = ["LSTM_largerBaseCNN", "LSTM_BaselineCNN"]
    stopping_models = ["Stopping_base", "Stop_AuxBaselineCNN", "MotionImage_attention"]
    if model_name in LSTM_models:
        return test.test_eeVel_LSTM()
    elif model_name in stopping_models:
        if use_saved_locations:
            return test.test_eeVelGrasp_savedPos(constrained)
        else:
            return test.test_eeVelGrasp(constrained)
    else:
        return test.test_eeVel(constrained)