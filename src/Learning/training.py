import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from .utils.utils_train import get_loss, get_optimiser, getReconProcessing

class Train():

    def __init__(self,
        model: torch.nn.Module,
        dataset: DataLoader,
        stopping_dataset: DataLoader,
        use_gpu: bool,
        epochs: int,
        stopping_epochs: int,
        batch_size: int,
        optimiser: str,
        lr: float,
        weight_decay: float = 1e-7,
        loss: str = 'MSE',
        stopping_loss: str = 'BCE',
        recon_size: int = 16
    ) -> None:

        self.epochs = epochs
        self.stopping_epochs = stopping_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.wd = weight_decay
        self.recon_size = recon_size

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            print("GPU acceleration enabled")
            print(f"Training on {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')

        self.model = model.to(device=self.device)
        self.dataloader = dataset
        self.stopping_dataloader = stopping_dataset

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

    def train_AE(self):
        print_every = 10
        dtype = torch.float32
        self.model.train()
        processing: T.Compose = getReconProcessing(self.recon_size)

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

    def train_auxStopIndividual(self):
        print_every = 10
        dtype = torch.float32
        self.model.train()

        print("\nInitiating Training for Reaching")
        for epoch in range(self.epochs):
            for t, (x, labels) in enumerate(self.dataloader):
                x = x.to(device=self.device, dtype=dtype)
                labels = torch.cat(labels, dim=1)
                labels = labels.to(device=self.device, dtype=dtype)

                out = self.model(x, train_stop=False)
                loss = self.loss(out, labels)

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                if t % print_every == 0:
                    print(f"Epoch: {epoch+1}, Iteration {t}, loss = {loss:.6f}")
            print("\n\n")

        print("\nReaching Training Ended\n")
        self.model.freeze_backbone()
        self.optimiser = optim.Adamax(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=self.wd
        )
        print("\nInitiating Training for Stopping")

        for name, param in self.model.named_parameters():
            if param.requires_grad:print(name)

        for epoch in range(self.stopping_epochs):
            for t, (x, stop_label) in enumerate(self.stopping_dataloader):

                x = x.to(device=self.device, dtype=dtype)
                stop_label = stop_label.to(device=self.device, dtype=dtype)

                out = self.model(x, train_stop=True)
                loss = self.stopping_loss(out, stop_label)

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                if t % print_every == 0:
                    print(f"Epoch: {epoch+1}, Iteration {t}, loss = {loss:.6f}")
            print("\n\n")

        for name, param in self.model.named_parameters():
            if param.requires_grad:print(name)

    
