from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image

class TL_onlyStop(Dataset):
    def __init__(
        self, 
        dataset_path,
        transform = None,
        filter_stop: bool=False
    ) -> None:

        self.df = pd.read_csv(dataset_path + "data.csv")
        self.transform = transform
        self.dataset_path = dataset_path

        if filter_stop:
            self.filter_stopData()


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df["imLoc"][index]
        stop = np.array([self.df['stop'][index]], dtype=float)

        image = Image.open(self.dataset_path + filename)
        if self.transform is not None:
            image = self.transform(image)
        return image, stop

    def filter_stopData(self):
        self.df = self.df.drop(self.df[self.df.loc[:,"stop"] == 1].index)
        self.df.reset_index(inplace=True)