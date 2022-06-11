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

#IMPORTANT GENERAL STUFF
EPOCHS = 100
BATCH_SIZE = 64
LR = 0.001
WD = 1e-7
USE_GPU = True
PATH_DATASET = "../../Demos/Dataset/try_2/"


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
class SimDataset(Dataset):
    def __init__(self, csv_path, transform = None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df["imLoc"][index]
        jointVel = [float(item) for item in self.df['jVel'][index].split(",")]
        eePos = [float(item) for item in self.df['eePos'][index].split(",")]
        cPos = [float(item) for item in self.df['cPos'][index].split(",")]
        ee_target = [float(item) for item in self.df['ee_target'][index].split(",")]

        # #push them into a single array so that we can output them without much issue here. MIGHT WANT TO CHANGE THIS LATER!!!
        jointVel.extend(eePos)
        jointVel.extend(cPos)


        image = Image.open(PATH_DATASET + filename)
        if self.transform is not None:
            image = self.transform(image)
        return image, np.array(ee_target)
        # return image, jointVel,eePos,cPos


trainSet = SimDataset(PATH_DATASET + "data.csv", transform)
trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, num_workers=0)

# ---------------- Train ---------------- #
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("Using GPU to train on")
else:
    device = torch.device('cpu')

print_every = 10
dtype = torch.float32
def train_model(model,optimizer,epochs=1):
    model = model.to(device=device)
    mseLoss = nn.MSELoss()
    for e in range(epochs):
        for t, (x, ee_v) in enumerate(trainLoader):
            model.train()
            x = x.to(device=device,dtype=dtype)
            ee_v = ee_v.to(device=device,dtype=dtype)

            out = model(x)
            loss = mseLoss(out, ee_v)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if t % print_every == 0:
                print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))



# ---------------- Run Training ---------------- #
torch.cuda.empty_cache()
model = BaselineCNN()
optimizer = optim.Adamax(model.parameters(), lr=LR, weight_decay=WD)

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters is: {}".format(params))
train_model(model, optimizer, epochs = EPOCHS)

# save the model
torch.save(model.state_dict(), 'TrainedModels/model.pt')