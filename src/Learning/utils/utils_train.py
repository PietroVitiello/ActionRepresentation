import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as T
from PIL import Image, ImageOps

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
    else:
        raise Exception("The proposed loss is not available. Try a different one!")

def getReconProcessing(recon_size):
    processing = T.Compose([
        T.Grayscale(),
        T.Resize((recon_size, recon_size))
    ])
    return processing