from os.path import dirname, join, abspath
from typing import List
import numpy as np
import pandas as pd
import yaml

import torch
import torchvision.transforms as T

from pyrep import PyRep

def get_run_id(model_name: str, config_filename: str):
    with open(f"Learning/TrainedModels/{config_filename}.yaml", 'r') as file:
        configs = yaml.safe_load(file)
    model_data = configs[model_name]
    return model_data["id"]

def get_metrics(model_name: str, config_filename: str):
    with open(f"Learning/TrainedModels/{config_filename}.yaml", 'r') as file:
        configs = yaml.safe_load(file)
    model_data = configs[model_name]

    in_metrics = model_data["input_metrics"]
    eeVel_metrics = model_data["eeVel_metrics"]
    try:
        recon_metrics = model_data["recon_metrics"]
    except KeyError:
        recon_metrics = None
    return in_metrics, eeVel_metrics, recon_metrics

class undo_dataTransform():
    def __init__(self, mean: List[float], std: List[float]) -> None:
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    
    def __call__(self, x):
        return (x * self.std) + self.mean

def undo_imageTransform(mean: List[float], std: List[float]):
    std = 1/torch.Tensor(std)
    mean = -torch.Tensor(mean)
    transform = T.Compose(
            [
                T.Normalize([0]*len(mean), std.tolist()),
                T.Normalize(mean.tolist(), [1]*len(mean))
            ]
        )
    return transform

def get_transform(use_metrics, model_name, config_filename):
    if use_metrics is False:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        #need to transform and need to normalize after
        in_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean, std)
            ]
        )
        return in_transform, None, None

    else:
        metrics = get_metrics(model_name, config_filename)
        mean_in, std_in = metrics[0]
        in_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean_in, std_in)
            ]
        )

        mean_eeVel, std_eeVel = metrics[1]
        eeVel_unnorm = undo_dataTransform(mean_eeVel, std_eeVel)
        try:
            mean_recon, std_recon = metrics[2]
            recon_unnorm = undo_imageTransform(mean_recon, std_recon)
        except:
            recon_unnorm = None

        return in_transform, eeVel_unnorm, recon_unnorm
