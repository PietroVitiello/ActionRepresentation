import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from .train import Training
from ..utils.utils_train import get_loss, get_optimiser, getReconProcessing, undoTransform
# from .utils.utils_dataloader import undoTransform
import wandb

class Train_eeVelAux_wandb(Training):

    def __init__(self,
        model: torch.nn.Module,
        model_name: str,
        dataset: DataLoader,
        val_dataset: DataLoader,
        transform: T,
        use_gpu: bool,
        epochs: int,
        batch_size: int,
        optimiser: str,
        lr: float,
        weight_decay: float = 1e-7,
        loss: str = 'MSE',
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

        self.input_transform, self.recon_transform, self.output_transform = transform

        print("\nEstablishing connection with Weights and Biases")
        self.run = wandb.init(
            project="New-Robot-Action-Representation",
            reinit=True,
            tags=["conv2fc"]
        )
        self.run.name = model_name
        self.run.save()
        self._wandb_config(optimiser, loss)

    def _wandb_config(self, optimiser, loss):
        self.run.config.epochs = self.epochs
        self.run.config.batch_size = self.batch_size
        self.run.config.lr = self.lr
        self.run.config.weight_decay = self.wd
        self.run.config.optimiser = optimiser
        self.run.config.loss = loss

    def _wandb_log_epoch(
        self,
        epoch,
        loss,
        val_loss
    ):
        self.run.log({
            "epoch": epoch,
            "loss": loss,
            "val_loss": val_loss
        })

    def train(self):
        print_every = 10
        dtype = torch.float32
        self.model.train()

        for epoch in range(self.epochs):
            epoch_loss = 0
            for t, (x, labels) in enumerate(self.dataloader):

                x = x.to(device=self.device,dtype=dtype)
                labels = torch.cat(labels, dim=1)
                labels = labels.to(device=self.device,dtype=dtype)

                out = self.model(x)
                loss = self.loss(out, labels)

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                epoch_loss += loss

                if t % print_every == 0:
                    print(f"Epoch: {epoch+1}, Iteration {t}, loss = {loss:.6f}")

            val_loss = self.val_reaching()
            self._wandb_log_epoch(epoch, epoch_loss/(t+1), val_loss)
            print("\n\n")

        self.run.finish()

        

    
