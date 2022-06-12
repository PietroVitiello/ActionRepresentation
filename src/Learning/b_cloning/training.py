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

from models import BaselineCNN

class Train():

    def __init__(self,
        model: torch.nn.Module,
        dataset: DataLoader,
        epochs = 100,
        batch_size = 64,
        optimiser = 'Adamax',
        lr = 0.001,
        weight_decay = 1e-7,
        loss = 'MSE',
        use_gpu = True
    ) -> None:

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.wd = weight_decay

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            print("Using GPU to train on")
        else:
            self.device = torch.device('cpu')

        self.model = model.to(device=self.device)
        self.dataloader = dataset

        # OPTIMISER
        if optimiser == 'Adamax':
            self.optimiser = optim.Adamax(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise Exception("The proposed optimiser is not available. Try a different one!")
        
        # LOSS
        if loss == 'MSE':
            self.loss = nn.MSELoss()
        else:
            raise Exception("The proposed loss is not available. Try a different one!")

    def train_eeVel(self):

        print_every = 10
        dtype = torch.float32
        self.model.train()

        for epoch in range(self.epochs):
            print("\n\n")
            for t, (x, ee_v) in enumerate(self.dataloader):
                x = x.to(device=self.device,dtype=dtype)
                ee_v = ee_v.to(device=self.device,dtype=dtype)

                out = self.model(x)
                loss = self.loss(out, ee_v)

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                if t % print_every == 0:
                    print('Epoch: %d, Iteration %d, loss = %.4f' % (epoch+1, t, loss.item()))
