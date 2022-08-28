from typing import List
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from .train import Training
from ..utils.utils_train import get_loss, get_optimiser, getReconProcessing, undoTransform
from ..utils.utils_test import undo_imageTransform
# from .utils.utils_dataloader import undoTransform

class Train_AuxStop(Training):

    def __init__(self,
        model: torch.nn.Module,
        model_name: str,
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
        recon_size: int = 16,
        tags: List[str] = None
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

        self.input_transform, _, self.output_transform = transform

        self.stopping_epochs = stopping_epochs
        self.stopping_dataloader = stopping_dataset
        self.stopping_loss = get_loss(stopping_loss)
        self.stopping_optimiser = None

        print("\nEstablishing connection with Weights and Biases")
        config = self._wandb_config(optimiser, loss)
        if tags is not None:
            self.run = wandb.init(
                project="New-Robot-Action-Representation",
                reinit=True,
                config=config,
                tags=tags
            )
        else:
            self.run = wandb.init(
                project="New-Robot-Action-Representation",
                reinit=True,
                config=config
            )
        self.run.name = model_name

    def _wandb_config(self, optimiser, loss):
        config = {
            "epochs" : self.epochs,
            "batch_size" : self.batch_size,
            "lr" : self.lr,
            "weight_decay" : self.wd,
            "optimiser" : optimiser,
            "loss" : loss
        }
        print("\n")
        return config

    def _wandb_log_epoch(
        self,
        epoch,
        loss,
        val_loss
    ):
        self.run.log({
            "epoch": epoch,
            "action_loss": loss,
            "val_action_loss": val_loss
        })

    def get_run_id(self):
        return self.run.id
        
    def train_reaching(self):
        print_every = 10
        dtype = torch.float32
        self.model.train()

        for epoch in range(self.epochs):
            epoch_loss = 0
            for t, (x, labels) in enumerate(self.dataloader):
                x = x.to(device=self.device,dtype=dtype)
                x = self.input_transform(x)

                labels = torch.cat(labels, dim=1)
                labels = labels.to(device=self.device,dtype=dtype)

                out = self.model(x, train_stop=False)
                loss = self.loss(out, labels)

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                epoch_loss += loss

                if t % print_every == 0:
                    print(f"Epoch: {epoch+1}, Iteration {t}, loss = {loss:.6f}")

            val_loss = self.val_reaching(train_stop=False)
            self._wandb_log_epoch(epoch, epoch_loss/(t+1), val_loss)
            print("\n\n")

        self.run.finish()

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

                self.stopping_optimiser.zero_grad()
                loss.backward()
                self.stopping_optimiser.step()

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
        print(filter(lambda p: p.requires_grad, self.model.parameters()))
        
        print("\nInitiating Training for Stopping")
        self.train_stopping()
        print("\nStopping Training Ended\n")

        

    
