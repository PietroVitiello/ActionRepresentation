import pandas as pd
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from .trainloader import SimDataset

class TL_aux(SimDataset):
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

        jointTarget = np.array([float(item) for item in self.df['j_targetVel'][index].split(",")])
        jointVel = np.array([float(item) for item in self.df['jVel'][index].split(",")])
        joint_pos = np.array([float(item) for item in self.df['jPos'][index].split(",")])
        eeTarget = np.array([float(item) for item in self.df['ee_targetVel'][index].split(",")])
        eeVel = np.array([float(item) for item in self.df['eeVel'][index].split(",")])
        eePos = np.array([float(item) for item in self.df['eePos'][index].split(",")])
        eeOri = np.array([float(item) for item in self.df['eeOri'][index].split(",")])
        cPos = np.array([float(item) for item in self.df['cPos'][index].split(",")])
        stop = np.array([self.df['stop'][index]], dtype=float)

        image = Image.open(self.dataset_path + filename)
        if self.transform is not None:
            image = self.transform(image)
        elif self.transform is None:
            image = T.ToTensor()(image)
        return image, (eeTarget, eePos, eeOri, cPos)

    def filter_stopData(self):
        self.df = self.df.drop(self.df[self.df.loc[:,"stop"] == 1].index)
        self.df.reset_index(inplace=True)

    def calculate_statistics(self):
        dataloader = DataLoader(self, batch_size=120, num_workers=1)
        for (x, labels) in dataloader:
            try:
                input_image = torch.cat((input_image, x), axis=0)
                data = torch.cat((data, torch.cat(labels, dim=1)), axis=0)
            except NameError:
                input_image = x
                data = torch.cat(labels, dim=1)
        input_std, input_mean = torch.std_mean(input_image, axis=[0,2,3])
        input_std = torch.ones(input_std.shape)
        input_mean = torch.zeros(input_mean.shape)
        print(f"input -->\tmean:{input_mean}, \tstd: {input_std}")
        data_std, data_mean = torch.std_mean(data, axis=0)
        data_std = torch.ones(data_std.shape)
        data_mean = torch.zeros(data_mean.shape)
        print(f"data -->\tmean:{data_mean}, \tstd: {data_std}\n\n")
        return (input_std, input_mean), (None, None), (data_std, data_mean)

    def get_transforms(self, get_stats: bool=False):
        def vector_transform(x: torch.Tensor):
            return (x - stats[2][1]) / stats[2][0]
        print("Calculating mean and standard deviation for data")
        stats = self.calculate_statistics()
        input_transform = T.Normalize(stats[0][1], stats[0][0])
        if get_stats == False:
            return input_transform, None, vector_transform
        else:
            return (input_transform, None, vector_transform), stats