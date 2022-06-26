import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from .utils.argParser_train import get_loss, get_optimiser, getReconProcessing

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
        stopping_loss = 'BCE',
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

        self.optimiser = get_optimiser(optimiser, self.model, self.lr, self.wd)
        self.loss = get_loss(loss)
        self.stopping_loss = get_loss(stopping_loss)

    def train_eeVel(self):
        print_every = 10
        dtype = torch.float32
        self.model.train()

        for epoch in range(self.epochs):
            print("\n\n")
            for t, (x, labels) in enumerate(self.dataloader):

                x = x.to(device=self.device,dtype=dtype)
                eeTarget = labels[0]
                eeTarget = eeTarget.to(device=self.device,dtype=dtype)

                out = self.model(x)
                loss = self.loss(out, eeTarget)

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                if t % print_every == 0:
                    print('Epoch: %d, Iteration %d, loss = %.6f' % (epoch+1, t, loss.item()))

    def train_eeVelAux(self):
        print_every = 10
        dtype = torch.float32
        self.model.train()

        for epoch in range(self.epochs):
            print("\n\n")
            for t, (x, labels) in enumerate(self.dataloader):

                x = x.to(device=self.device,dtype=dtype)
                labels = torch.cat(labels, dim=1)
                labels = labels.to(device=self.device,dtype=dtype)

                out = self.model(x)
                loss = self.loss(out, labels)

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                if t % print_every == 0:
                    print(f"Epoch: {epoch+1}, Iteration {t}, loss = {loss:.6f}")

    def train_AE(self, recon_size):
        print_every = 10
        dtype = torch.float32
        self.model.train()
        processing: T.Compose = getReconProcessing(recon_size)

        for epoch in range(self.epochs):
            print("\n\n")
            for t, (x, labels) in enumerate(self.dataloader):
                x = x.to(device=self.device,dtype=dtype)
                labels = torch.cat(labels, dim=1)
                labels = labels.to(device=self.device,dtype=dtype)
                recon_labels: torch.Tensor = processing(x)
                recon_labels = recon_labels.squeeze()

                recon_img, out = self.model(x)
                recon_loss = self.loss(recon_img, recon_labels)
                loss_out = self.loss(out, labels)
                loss = recon_loss + loss_out

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                if (t+1) % print_every == 0:
                    print(f"Epoch: {epoch+1}, Iteration {t+1}, loss = {loss:.6f}")

    def train_stopping(self):
        print_every = 10
        dtype = torch.float32
        self.model.train()

        for epoch in range(self.epochs):
            print("\n\n")
            for t, (x, labels) in enumerate(self.dataloader):

                x = x.to(device=self.device,dtype=dtype)
                labels = torch.cat(labels, dim=1)
                labels = labels.to(device=self.device,dtype=dtype)

                out = self.model(x)
                # print(out[:,0].shape)
                # print(out[:,:-1].shape)
                loss = self.loss(out[:,:-1], labels[:,:-1])
                stop_loss = self.stopping_loss(out[:,-1], labels[:,-1])
                loss = loss + stop_loss

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                if t % print_every == 0:
                    print(f"Epoch: {epoch+1}, Iteration {t}, loss = {loss:.6f}")

    
