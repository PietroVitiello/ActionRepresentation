from torch.utils.data import Dataset
import torchvision.transforms as T
import pandas as pd
import numpy as np
from PIL import Image

from .trainloader import SimDataset

class TL_onlyStop(SimDataset):
    def __init__(
        self, 
        dataset_path,
        transform = None,
        filter_stop: bool=False,
        considered_indices: np.ndarray = None
    ) -> None:

        super().__init__(
            dataset_path,
            transform,
            filter_stop,
            considered_indices
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df["imLoc"][index]
        stop = np.array([self.df['stop'][index]], dtype=float)

        image = Image.open(self.dataset_path + filename)
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)
        return image, stop

    def filter_stopData(self):
        self.df = self.df.drop(self.df[self.df.loc[:,"stop"] == 1].index)
        self.df.reset_index(inplace=True)