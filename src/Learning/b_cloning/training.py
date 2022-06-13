import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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

        self.optimiser = self.get_optimiser(optimiser)
        self.loss = self.get_loss(loss)

    def get_optimiser(self, optimiser):
        if optimiser == 'Adamax':
            return optim.Adamax(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        else:
            raise Exception("The proposed optimiser is not available. Try a different one!")

    def get_loss(self, loss):
        if loss == 'MSE':
            return nn.MSELoss()
        else:
            raise Exception("The proposed loss is not available. Try a different one!")

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
    
    def train_model(self, mode):
        if mode == 'eeVel':
            self.train_eeVel()
        elif mode == 'eeVel_aux':
            self.train_eeVelAux()
        else:
            raise Exception("Training modality selected has not been recognized")

    
