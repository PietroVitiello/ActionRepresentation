from trainloader import SimDataset
import pandas as pd
import numpy as np
from PIL import Image

class TL_eeVel(SimDataset):
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
        eeTarget = np.array([float(item) for item in self.df['ee_targetVel'][index].split(",")])

        image = Image.open(self.dataset_path + filename)
        if self.transform is not None:
            image = self.transform(image)
        return image, eeTarget

    def filter_stopData(self):
        self.df = self.df.drop(self.df[self.df.loc[:,"stop"] == 1].index)
        self.df.reset_index(inplace=True)