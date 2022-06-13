import torch
from torch.nn import Conv2d, MaxPool2d
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import sampler
from torchvision import datasets, transforms
from os.path import dirname, join, abspath
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import PIL
import PIL.Image as Image
import random

from .models import BaselineCNN, Aux_BaselineCNN, LSTM_BaselineCNN, LSTM_largerBaseCNN
from trainloader import SimDataset
from .training import Train

def model_choice(model_name, *args):
    if model_name == "BaselineCNN":
        return BaselineCNN(*args)
    elif model_name == "Aux_BaselineCNN":
        return Aux_BaselineCNN(*args)
    elif model_name == "LSTM_BaselineCNN":
        return LSTM_BaselineCNN(*args)
    elif model_name == "LSTM_largerBaseCNN":
        return LSTM_largerBaseCNN(*args)
    else:
        raise Exception("There is no such model available")

def behaviouralCloning_training(
    data_folder,
    saved_model_name,
    model_name = "BaselineCNN",
    epochs = 100,
    batch_size = 64,
    lr = 0.001,
    weight_decay = 1e-7,
    optimiser = 'Adamax',
    loss = 'MSE',
    training_method = 'eeVel',
    use_gpu = True
):
    dataset_path = join(dirname(abspath(__file__)), f"../../Demos/Dataset/{data_folder}/")
    
    #setup image transforms
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])

    #need to transform and need to normalize after
    transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean.tolist(), std.tolist())
            ]
        )

    # ---------------- Dataset ---------------- #
    trainSet = SimDataset(dataset_path, transform)
    trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=False, num_workers=1)

    # ---------------- Training ---------------- #
    torch.cuda.empty_cache()
    model = model_choice(model_name, 9, 3)
    training = Train(model, trainLoader, epochs, batch_size, optimiser, lr, weight_decay, loss, use_gpu)
    
    training.train_model(training_method)

    # save the model
    save_dir = join(dirname(abspath(__file__)), f'TrainedModels/{saved_model_name}.pt')
    torch.save(model.state_dict(), save_dir)










#IMPORTANT GENERAL STUFF
# EPOCHS = 100
# BATCH_SIZE = 64
# LR = 0.001
# WD = 1e-7
# USE_GPU = True
# PATH_DATASET = "../../Demos/Dataset/followDummy_1/"


# #setup image transforms
# mean = torch.Tensor([0.485, 0.456, 0.406])
# std = torch.Tensor([0.229, 0.224, 0.225])

# #need to transform and need to normalize after
# transform = transforms.Compose(
#         [
#             transforms.ToTensor(),
#             transforms.Normalize(mean.tolist(), std.tolist())
#         ]
#     )

# # ---------------- Dataset ---------------- #
# trainSet = SimDataset(PATH_DATASET, transform)
# trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, num_workers=1)

# # ---------------- Train ---------------- #
# if USE_GPU and torch.cuda.is_available():
#     device = torch.device('cuda:0')
#     print("Using GPU to train on")
# else:
#     device = torch.device('cpu')

# print_every = 10
# dtype = torch.float32
# def train_model(model,optimizer,epochs=1):
#     model = model.to(device=device)
#     mseLoss = nn.MSELoss()
#     for e in range(epochs):
#         for t, (x, ee_v) in enumerate(trainLoader):
#             model.train()
#             x = x.to(device=device,dtype=dtype)
#             ee_v = ee_v.to(device=device,dtype=dtype)

#             out = model(x)
#             loss = mseLoss(out, ee_v)

#             optimizer.zero_grad()

#             loss.backward()

#             optimizer.step()

#             if t % print_every == 0:
#                 print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))



# ---------------- Run Training ---------------- #
# torch.cuda.empty_cache()
# model = BaselineCNN(3)
# optimizer = optim.Adamax(model.parameters(), lr=LR, weight_decay=WD)

# params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print("Total number of parameters is: {}".format(params))
# train_model(model, optimizer, epochs = EPOCHS)

# # save the model
# torch.save(model.state_dict(), 'TrainedModels/baselineCNN_follow.pt')