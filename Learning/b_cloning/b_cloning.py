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

#IMPORTANT GENERAL STUFF
EPOCHS = 100
BATCH_SIZE = 64
LR = 0.001
WD = 1e-7
USE_GPU = True


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

#Create the dataloader. You may want to create a validation and training set for later too.
class SimDataset(Dataset):
    def __init__(self,csv_path,transform = None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df["imLoc"][index]
        jointVel = [float(item) for item in self.df['jVel'][index].split(",")]
        eePos = [float(item) for item in self.df['eePos'][index].split(",")]
        cPos = [float(item) for item in self.df['cPos'][index].split(",")]

        # #push them into a single array so that we can output them without much issue here. MIGHT WANT TO CHANGE THIS LATER!!!
        jointVel.extend(eePos)
        jointVel.extend(cPos)


        image = Image.open(filename)
        if self.transform is not None:
            image = self.transform(image)
        return image,np.array(jointVel)
        # return image, jointVel,eePos,cPos


trainSet = SimDataset("lol.csv",transform)
# print(len(trainSet[0][1])+len(trainSet[0][2])+len(trainSet[0][3]))
#May want to create a trainining and validation set for later

#dataset loader - for each sample, 0 gives image, 1 gives joint vels, 2 gives eepos, 3 gives cPos
trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, num_workers=0)

#---------- MODEL SETUP -------------
class ResNet(nn.Module):
    def __init__(self,num_outputs = 13,fconv=[3,1,1]):
        super(ResNet,self).__init__()

        self.conv1 = nn.Sequential(Conv2d(3,64,kernel_size= fconv[0],stride=fconv[1],padding=fconv[2],bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(Conv2d(64,128,kernel_size=fconv[0],stride=fconv[1],padding=fconv[2],bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(Conv2d(128, 256, kernel_size=fconv[0], stride=fconv[1], padding=fconv[2], bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(inplace=True))
        self.fc = nn.Linear(256*8*8,num_outputs)

    def forward(self,x):

        x = self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)

        return x

#----- RUN THE MODEL -----
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

print_every = 10
dtype = torch.float32
def train_model(model,optimizer,epochs=1):
    model = model.to(device=device)
    mseLoss = nn.MSELoss()
    for e in range(epochs):
        for t, (x, jv) in enumerate(trainLoader):
        # for t, (x,jv, ep, cp) in enumerate(trainLoader):
            model.train()
            x = x.to(device=device,dtype=dtype)
            # jv.extend(ep)
            # jv.extend(cp)
            jv = jv.to(device=device,dtype=dtype)
            # ep = ep.to(device=device,dtype=torch.long)
            # cp = cp.to(device=device,dtype=torch.long)

            out = model(x)
            loss = mseLoss(out,jv)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if t % print_every == 0:
                print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))

torch.cuda.empty_cache()
# define and train the network
model = ResNet() #same sizing for both ResNet34 and 50 depending on the type of residual layer used
optimizer = optim.Adamax(model.parameters(), lr=LR, weight_decay=WD)

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters is: {}".format(params))

train_model(model, optimizer, epochs = EPOCHS)

# save the model
torch.save(model.state_dict(), 'model.pt')