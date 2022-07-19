from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

from ..Models.BaselineCNN.models import BaselineCNN, Aux_BaselineCNN, LSTM_BaselineCNN, LSTM_largerBaseCNN
from ..Models.AutoEncoder.models import SpatialAE_fc, StrengthSpatialAE_fc
from ..Models.Stopping.models import Stopping_base, Stop_AuxBaselineCNN

from ..training import Train
from ..testing import Test

def getTrainLoader(trainSet, batch_size, model) -> DataLoader:
    LSTM_models = ["LSTM_largerBaseCNN", "LSTM_BaselineCNN"]
    if model in LSTM_models:
        return DataLoader(trainSet, batch_size=batch_size, shuffle=False, num_workers=1)
    else:
        return DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=1)

def get_episodeNum(filename: str) -> int:
    _, ep_dir = filename.split("episode_")
    ep_num, _ = ep_dir.split("/", 1)
    return int(ep_num)

def get_runNum(filename: str) -> int:
    _, run_dir = filename.split("run_")
    run_num, _ = run_dir.split("/", 1)
    return int(run_num)

def get_stepNum(filename: str) -> int:
    _, step_dir = filename.split("step_")
    step_num, _ = step_dir.split("/", 1)
    return int(step_num)

def get_numEpisodeRun(filename: str) -> Tuple[int, int]:
    return get_episodeNum(filename), get_runNum(filename)
