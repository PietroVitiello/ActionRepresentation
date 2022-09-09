from typing import List, Tuple
import numpy as np
import yaml
from ruamel.yaml import YAML

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T

from ..Models.BaselineCNN.models import BaselineCNN, Aux_BaselineCNN, LSTM_BaselineCNN, LSTM_largerBaseCNN
from ..Models.AutoEncoder.models import SpatialAE_fc, StrengthSpatialAE_fc
from ..Models.Stopping.models import Stopping_base, Stop_AuxBaselineCNN

from ..training import Train
from ..Testing.testing import Test

ryaml = YAML()

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
    step_num, _ = step_dir.split(".", 1)
    return int(step_num)

def get_numEpisodeRun(filename: str) -> Tuple[int, int]:
    return get_episodeNum(filename), get_runNum(filename)

def getTransformation(mean: List[float], std: List[float]):
    transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean.tolist(), std.tolist())
            ]
        )
    return transform

def undoTransform(mean: List[float], std: List[float]):
    std = 1/std
    mean = -mean
    transform = T.Compose(
            [
                T.Normalize([0]*len(mean), std.tolist()),
                T.Normalize(mean.tolist(), [1]*len(mean))
            ]
        )
    return transform

def find_saved_stats(dataset_name: str):
    with open("Demos/Dataset/descriptions.yaml", 'r') as file:
        configs = yaml.safe_load(file)
    dataset_conf = configs[dataset_name]
    try:
        means = dataset_conf["means"]
        stds = dataset_conf["stds"]
        return means, stds
    except KeyError:
        return False

def save_stats(dataset_name: str, means: List[np.ndarray], stds: List[np.ndarray], data_names: List[str]):
    with open("Demos/Dataset/descriptions.yaml", 'r') as file:
        configs = ryaml.load(file)
    with open("Demos/Dataset/descriptions.yaml", 'w') as file:
        configs[dataset_name]["means"] = means
        configs[dataset_name]["stds"] = stds
        configs[dataset_name]["data_names"] = stds
        ryaml.dump(configs, file)

