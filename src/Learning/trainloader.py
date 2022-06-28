from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from PIL import Image

class SimDataset(Dataset):
    def __init__(
        self, 
        dataset_path,
        transform = None,
        dataset_mode: str="eeVel",
        filter_stop: bool=False
    ) -> None:

        self.df = pd.read_csv(dataset_path + "data.csv")
        self.transform = transform
        self.dataset_path = dataset_path
        self.dataset_mode = dataset_mode

        if filter_stop:
            self.filter_stopData()


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.dataset_mode == "eeVel":
            return self.getitem_eeVel(index)
        elif self.dataset_mode == "aux":
            return self.getitem_aux(index)
        elif self.dataset_mode == "stop":
            return self.getitem_stop(index)
        elif self.dataset_mode == "onlyStop":
            return self.getitem_onlyStop(index)
        else:
            raise Exception("The selected dataset mode is not supported")

    def getitem_eeVel(self, index):
        filename = self.df["imLoc"][index]
        eeTarget = np.array([float(item) for item in self.df['ee_targetVel'][index].split(",")])

        image = Image.open(self.dataset_path + filename)
        if self.transform is not None:
            image = self.transform(image)
        return image, eeTarget

    def getitem_aux(self, index):
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
        return image, (eeTarget, eePos, eeOri, cPos)

    def getitem_stop(self, index):
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
        return image, (eeTarget, eePos, eeOri, cPos, stop)

    def getitem_onlyStop(self, index):
        filename = self.df["imLoc"][index]
        stop = np.array([self.df['stop'][index]], dtype=float)

        image = Image.open(self.dataset_path + filename)
        if self.transform is not None:
            image = self.transform(image)
        return image, stop

    def filter_stopData(self):
        self.df = self.df.drop(self.df[self.df.loc[:,"stop"] == 1].index)