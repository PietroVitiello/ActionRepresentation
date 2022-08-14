from abc import abstractclassmethod, abstractmethod
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from ..utils.utils_train import get_loss, get_optimiser, getReconProcessing, undoTransform
# from train_eeVel import Train_eeVel
# from train_eeVelAux import Train_eeVelAux
# from train_AE import Train_AE
# from .utils.utils_dataloader import undoTransform

class Training():

    def __init__(self,
        model: torch.nn.Module,
        dataset: DataLoader,
        val_datasets: Tuple[DataLoader],
        use_gpu: bool,
        epochs: int,
        batch_size: int,
        optimiser: str,
        lr: float,
        weight_decay: float = 1e-7,
        loss: str = 'MSE',
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.wd = weight_decay

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            print("GPU acceleration enabled")
            print(f"Training on {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')

        self.model = model.to(device=self.device)
        self.dataloader = dataset
        self.val_dataloaders = val_datasets
        self.input_transform, self.recon_transform, self.output_transform = None, None, None

        self.optimiser = get_optimiser(optimiser, self.model, self.lr, self.wd)
        self.loss = get_loss(loss)

    # def __init__(self,
    #     model: torch.nn.Module,
    #     dataset: DataLoader,
    #     stopping_dataset: DataLoader,
    #     transform: T,
    #     use_gpu: bool,
    #     epochs: int,
    #     stopping_epochs: int,
    #     batch_size: int,
    #     optimiser: str,
    #     lr: float,
    #     weight_decay: float = 1e-7,
    #     loss: str = 'MSE',
    #     stopping_loss: str = 'BCE',
    #     recon_size: int = 16
    # ) -> None:

    #     self.epochs = epochs
    #     self.stopping_epochs = stopping_epochs
    #     self.batch_size = batch_size
    #     self.lr = lr
    #     self.wd = weight_decay
    #     self.recon_size = recon_size

    #     if use_gpu and torch.cuda.is_available():
    #         self.device = torch.device('cuda:0')
    #         print("GPU acceleration enabled")
    #         print(f"Training on {torch.cuda.get_device_name()}")
    #     else:
    #         self.device = torch.device('cpu')

    #     self.model = model.to(device=self.device)
    #     self.dataloader = dataset
    #     self.stopping_dataloader = stopping_dataset
    #     self.transform = transform

    #     self.optimiser = get_optimiser(optimiser, self.model, self.lr, self.wd)
    #     self.loss = get_loss(loss)
    #     self.stopping_loss = get_loss(stopping_loss)

    def val_reaching_AE(self):
        def get_loss(x, label):
            loss = loss_fn(x, label)
            return torch.sum(loss) / torch.numel(loss) * loss.shape[0]

        val_dataloader = self.val_dataloaders[0]
        # self.model.eval()
        dtype = torch.float32
        loss_fn = nn.MSELoss(reduction='none')
        n_datapoints = len(val_dataloader.dataset)
        recon_loss = 0
        action_loss = 0
        with torch.no_grad():
            for t, (x, labels) in enumerate(val_dataloader):
                x = x.to(device=self.device, dtype=dtype)
                x = self.input_transform(x)

                recon_label = labels[-1]
                recon_label = recon_label.to(device=self.device, dtype=dtype)
                recon_label = self.recon_transform(recon_label)

                labels = torch.cat(labels[:-1], dim=1)
                labels = self.output_transform(labels)
                labels = labels.to(device=self.device, dtype=dtype)

                out, recon = self.model(x, train_stop=False)
                recon_loss += get_loss(recon, recon_label) / n_datapoints
                action_loss += get_loss(out, labels) / n_datapoints

        self.model.train(True)
        print(f"**Validation**\t recon_loss = {recon_loss:.6f}, action_loss = {action_loss:.6f}, loss = {recon_loss+action_loss:.6f}")
        return recon_loss, action_loss, recon_loss+action_loss

    def val_stopping(self):
        val_dataloader = self.val_dataloaders[1]
        dtype = torch.float32
        # loss_fn = nn.BCELoss(reduction='sum')
        n_datapoints = len(val_dataloader.dataset)
        # loss = 0
        correct_stop = 0
        correct_move = 0
        accuracy = 0
        with torch.no_grad():
            for t, (x, labels) in enumerate(val_dataloader):
                x = x.to(device=self.device, dtype=dtype)
                stop_label = labels.to(device=self.device, dtype=dtype)

                out = self.model(x, train_stop=True)
                out = torch.tensor(out > 0.96).detach().type(torch.float)

                # loss = loss_fn(out, stop_label)
                accuracy += torch.sum(torch.tensor(stop_label ==  out)) / n_datapoints
                correct_stop += sum(out[stop_label == 1]) / n_datapoints
                correct_move += sum((1 - out[stop_label == 0])) / n_datapoints

        self.model.train(True)
        print(f"**Validation**\t correct_move = {correct_move*100:.1f}%, correct_stop = {correct_stop*100:.1f}%, accuracy = {accuracy*100:.1f}%")

    @abstractclassmethod
    def train(self):
        pass

    # @staticmethod
    # def get_trainer(
    #     training_type: str,
    #     model: torch.nn.Module,
    #     model_name: str,
    #     dataset: DataLoader,
    #     val_datasets: Tuple[DataLoader],
    #     stopping_dataset: DataLoader,
    #     transform: T,
    #     use_gpu: bool,
    #     epochs: int,
    #     stopping_epochs: int,
    #     batch_size: int,
    #     optimiser: str,
    #     lr: float,
    #     weight_decay: float = 1e-7,
    #     loss: str = 'MSE',
    #     stopping_loss: str = 'BCE',
    #     recon_size: int = 16
    # ):
    #     if training_type == 'eeVel':
    #         Train_eeVel(
    #             model,
    #             dataset,
    #             use_gpu,
    #             epochs,
    #             batch_size,
    #             optimiser,
    #             lr,
    #             weight_decay,
    #             loss
    #         )
    #     elif training_type == 'eeVel_aux':
    #         Train_eeVelAux(
    #             model,
    #             dataset,
    #             use_gpu,
    #             epochs,
    #             batch_size,
    #             optimiser,
    #             lr,
    #             weight_decay,
    #             loss
    #         )
    #     elif training_type == 'AE':
    #         Train_AE(
    #             model,
    #             dataset,
    #             val_datasets,
    #             stopping_dataset,
    #             transform,
    #             use_gpu,
    #             epochs,
    #             stopping_epochs,
    #             batch_size,
    #             optimiser,
    #             lr,
    #             weight_decay,
    #             loss,
    #             stopping_loss,
    #             recon_size
    #         )
    #     elif training_type == 'wandb':
    #         Train_AE(
    #             model,
    #             model_name,
    #             dataset,
    #             val_datasets,
    #             stopping_dataset,
    #             transform,
    #             use_gpu,
    #             epochs,
    #             stopping_epochs,
    #             batch_size,
    #             optimiser,
    #             lr,
    #             weight_decay,
    #             loss,
    #             stopping_loss,
    #             recon_size
    #         )
    #     # elif training_type == 'stop':
    #     #     train.train_stopping()
    #     # elif training_type == 'aux_stopIndividual':
    #     #     train.train_auxStopIndividual()
    #     # elif training_type == 'motion_image':
    #     #     train.train_MotionImage()

    #     # elif training_type == 'pureAE':
    #     #     train.train_PureAE()
    #     else:
    #         raise Exception("Training modality selected has not been recognized")

        

    
