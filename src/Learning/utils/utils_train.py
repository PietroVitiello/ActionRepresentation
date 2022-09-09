from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as T
from PIL import Image, ImageOps

from ..Losses.directional import DirectionalLoss

def get_optimiser(
        optimiser,
        model: nn.Module,
        lr,
        wd
    ):
    if optimiser == 'Adamax':
        return optim.Adamax(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise Exception("The proposed optimiser is not available. Try a different one!")

def get_loss(loss) -> nn.modules.loss:
    if loss == 'MSE':
        return nn.MSELoss()
    if loss == 'BCE':
        return nn.BCELoss()
    if loss == 'directional':
        return DirectionalLoss()
    else:
        raise Exception("The proposed loss is not available. Try a different one!")

def getReconProcessing(recon_size):
    processing = T.Compose([
        T.Grayscale(),
        T.Resize((recon_size, recon_size))
    ])
    return processing

def undoTransform(mean: List[float], std: List[float]):
    # ones = torch.ones(len(std))
    std = torch.div(1, torch.tensor(std))
    mean = -torch.tensor(mean)
    transform = T.Compose(
            [
                T.Normalize([0]*len(mean), std.tolist()),
                T.Normalize(mean.tolist(), [1]*len(mean))
            ]
        )
    return transform