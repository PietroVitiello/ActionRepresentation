from typing import List
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from .train import Training
from ..Models.MotionIMG.models import MI_Net_indepAE
from ..utils.utils_train import get_loss, get_optimiser, getReconProcessing, undoTransform
from ..utils.utils_test import undo_imageTransform
# from .utils.utils_dataloader import undoTransform

class Train_AE_indep_wandb(Training):

    def __init__(self,
        model: MI_Net_indepAE,
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
        self.recon_size = recon_size
        self.input_transform, self.recon_transform, self.output_transform = transform

        self.optimiser = None

        self.ae_epochs = 60
        self.ae_loss = nn.MSELoss()
        self.ae_optimiser = None

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
        self.run.save()

    def _wandb_config(self, optimiser, loss):
        config = {
            "epochs" : self.epochs,
            "batch_size" : self.batch_size,
            "lr" : self.lr,
            "weight_decay" : self.wd,
            "optimiser" : optimiser,
            "loss" : loss
        }
        return config

    def _wandb_log_epoch(
        self,
        epoch,
        recon_loss,
        action_loss,
        loss,
        val_recon_loss,
        val_action_loss,
        val_loss
    ):
        self.run.log({
            "epoch": epoch,
            "recon_loss": recon_loss,
            "action_loss": action_loss,
            "loss": loss,
            "val_recon_loss": val_recon_loss,
            "val_action_loss": val_action_loss,
            "val_loss": val_loss
        })

    def _wandb_log_image_predictions(self, n_preds: int):
        mean = self.recon_transform.mean.tolist()
        std = self.recon_transform.std.tolist()
        unnorm = undo_imageTransform(mean, std)
        labels, preds = self.val_image_pred(n_preds)
        preds = unnorm(preds)
        for (label, pred) in zip(labels, preds):
            self.run.log({
                "label_reconstruction": wandb.Image(label),
                "predicted_reconstruction": wandb.Image(pred)
            })

    def train_AE(self):
        print_every = 10
        dtype = torch.float32

        for epoch in range(self.ae_epochs):
            epoch_recon_loss = 0
            for t, (x, labels) in enumerate(self.dataloader):
                x = x.to(device=self.device, dtype=dtype)
                x = self.input_transform(x)

                mi_label = labels[-1]
                mi_label = mi_label.to(device=self.device, dtype=dtype)
                mi_label = self.recon_transform(mi_label)

                mi = self.model(x, train_actions=False, train_stop=False)
                recon_loss = self.ae_loss(mi, mi_label)

                self.ae_optimiser.zero_grad()
                recon_loss.backward()
                self.ae_optimiser.step()

                #For logging purposes
                epoch_recon_loss += recon_loss

                if t % print_every == 0:
                    print(f"Epoch: {epoch+1:3d}, Iteration {t:4d}, recon_loss = {recon_loss:.6f}")
            val_recon_loss, val_action_loss, val_loss = self.val_reaching_AE()
            # print(epoch_recon_loss/(t+1))
            # print(epoch_action_loss/(t+1))
            # print(epoch_loss/(t+1))
            self._wandb_log_epoch(epoch, epoch_recon_loss/(t+1), None, None,
                                  val_recon_loss, val_action_loss, val_loss)
            print("\n\n")
        
    def train_reaching(self):
        print_every = 10
        dtype = torch.float32

        for epoch in range(self.epochs):
            epoch_action_loss = 0
            for t, (x, labels) in enumerate(self.dataloader):
                x = x.to(device=self.device, dtype=dtype)
                x = self.input_transform(x)

                labels = torch.cat(labels[:-1], dim=1)
                labels = self.output_transform(labels)
                labels = labels.to(device=self.device, dtype=dtype)

                out = self.model(x, train_stop=False)
                action_loss = self.loss(out, labels)

                self.optimiser.zero_grad()
                action_loss.backward()
                self.optimiser.step()

                #For logging purposes
                epoch_action_loss += action_loss

                if t % print_every == 0:
                    print(f"Epoch: {epoch+1:3d}, Iteration {t:4d}, action_loss = {action_loss:.6f}")
            val_recon_loss, val_action_loss, val_loss = self.val_reaching_AE()
            # print(epoch_recon_loss/(t+1))
            # print(epoch_action_loss/(t+1))
            # print(epoch_loss/(t+1))
            self._wandb_log_epoch(epoch, None, epoch_action_loss/(t+1), None,
                                  val_recon_loss, val_action_loss, val_loss)
            print("\n\n")

        self._wandb_log_image_predictions(n_preds=3)
        self.run.finish()

    def freeze_reaching(self):
        self.model.freeze_reach()
        self.ae_optimiser = optim.Adamax(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=self.wd
        )

    def unfreeze_reaching(self):
        self.model.unfreeze_reach()
        self.optimiser = optim.Adamax(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=self.wd
        )

    def freeze_backbone(self):
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
        self.freeze_reaching()

        print("\nInitiating Training for AutoEncoder")
        self.train_AE()
        print("\nReaching Training Ended\n")

        self.unfreeze_reaching()

        print("\nInitiating Training for Reaching")
        self.train_reaching()
        print("\nReaching Training Ended\n")

        self.freeze_backbone()
        
        print("\nInitiating Training for Stopping")
        self.train_stopping()
        self.val_reaching_AE()
        print("\nStopping Training Ended\n")

        

    
