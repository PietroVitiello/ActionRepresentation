import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from .train import Training
from ..utils.utils_train import get_loss, get_optimiser, getReconProcessing, undoTransform
# from .utils.utils_dataloader import undoTransform

class Train_AE(Training):

    def __init__(self,
        model: torch.nn.Module,
        dataset: DataLoader,
        val_dataset: DataLoader,
        stopping_dataset: DataLoader,
        transform: T,
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

        super().__init__(
            model,
            dataset,
            val_dataset,
            use_gpu,
            epochs,
            batch_size,
            optimiser,
            lr,
            weight_decay,
            loss
        )
        self.recon_size = recon_size
        self.input_transform, self.recon_transform, self.output_transform = transform

        self.stopping_epochs = stopping_epochs
        self.stopping_dataloader = stopping_dataset
        self.stopping_loss = get_loss(stopping_loss)
        self.stopping_optimiser = None

    # def output_transform(self, x: torch.Tensor):
    #     x = x.unsqueeze(dim=-1)
    #     print(x.shape)
    #     x = self._output_transform(x)
    #     return x.squeeze()

    def train_reaching(self):
        print_every = 10
        dtype = torch.float32

        print("\nInitiating Training for Reaching")
        for epoch in range(self.epochs):
            for t, (x, labels) in enumerate(self.dataloader):
                x = x.to(device=self.device, dtype=dtype)
                x = self.input_transform(x)

                mi_label = labels[-1]
                mi_label = mi_label.to(device=self.device, dtype=dtype)
                mi_label = self.recon_transform(mi_label)

                labels = torch.cat(labels[:-1], dim=1)
                labels = self.output_transform(labels)
                labels = labels.to(device=self.device, dtype=dtype)

                out, mi = self.model(x, train_stop=False)
                recon_loss = self.loss(mi, mi_label)
                action_loss = self.loss(out, labels)
                loss = action_loss + recon_loss

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                if t % print_every == 0:
                    print(f"Epoch: {epoch+1:3d}, Iteration {t:4d}, recon_loss = {recon_loss:.6f}, action_loss = {action_loss:.6f}, loss = {loss:.6f}")
            self.val_reaching_AE()
            print("\n\n")

    def freeze_reaching(self):
        self.model.freeze_backbone()
        self.stopping_optimiser = optim.Adamax(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=self.wd
        )

    def train_stopping(self):
        print_every = 10
        dtype = torch.float32

        for epoch in range(self.stopping_epochs):
            for t, (x, stop_label) in enumerate(self.stopping_dataloader):
                x = x.to(device=self.device, dtype=dtype)
                x = self.input_transform(x)
                stop_label = stop_label.to(device=self.device, dtype=dtype)

                out = self.model(x, train_stop=True)
                loss = self.stopping_loss(out, stop_label)

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                if t % print_every == 0:
                    print(f"Epoch: {epoch+1}, Iteration {t}, loss = {loss:.6f}")
            self.val_stopping()
            print("\n\n")

    def train(self):
        self.model.train(True)

        print("\nInitiating Training for Reaching")
        self.train_reaching()
        print("\nReaching Training Ended\n")

        self.freeze_reaching()
        
        print("\nInitiating Training for Stopping")
        self.train_stopping()
        print("\nStopping Training Ended\n")

        

    
