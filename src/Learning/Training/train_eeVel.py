import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from train import Training
from ..utils.utils_train import get_loss, get_optimiser, getReconProcessing, undoTransform
# from .utils.utils_dataloader import undoTransform

class Train_eeVel(Training):

    def __init__(self,
        model: torch.nn.Module,
        dataset: DataLoader,
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
            transform,
            use_gpu,
            epochs,
            batch_size,
            optimiser,
            lr,
            weight_decay,
            loss
        )

    def train(self):
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

        

    
