from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import pandas as pd
import numpy as np
from PIL import Image

from .trainloader import SimDataset
from ..utils.motion_image import get_motionImage
from ..utils.utils_dataloader import get_numEpisodeRun, get_stepNum

class TL_futureImage(SimDataset):
    def __init__(
        self, 
        dataset_path,
        transform = None,
        filter_stop: bool = False,
        delta_steps: int = 5,
        considered_indices: np.ndarray = None
    ) -> None:

        super().__init__(
            dataset_path,
            transform,
            filter_stop,
            considered_indices
        )
        
        self.cleaned_df = self.remove_unusable_data(self.df)
        self.delta_steps = delta_steps
        self.future_delta_target = delta_steps

    def __len__(self):
        # l_df = len(self.df)
        # last_filename = self.df.iloc[-1]["imLoc"]
        # ep_num, run_num = get_numEpisodeRun(last_filename)
        # n_demos = (ep_num+1) * (run_num+1)
        # # return l_df - (n_demos * self.delta_steps)
        # print(l_df - n_demos)
        # return l_df - n_demos
        return len(self.cleaned_df)

    def __getitem__(self, index):
        # print(index)
        index = self.cleaned_df[index]
        # print(index, "\n\n")
        filename, future_filename = self.get_valid_imagePair(index)

        eeTarget = np.array([float(item) for item in self.df['ee_targetVel'][index].split(",")])
        eePos = np.array([float(item) for item in self.df['eePos'][index].split(",")])
        eeOri = np.array([float(item) for item in self.df['eeOri'][index].split(",")])
        cPos = np.array([float(item) for item in self.df['cPos'][index].split(",")])

        image = Image.open(self.dataset_path + filename)
        future_image = Image.open(self.dataset_path + future_filename)

        if self.transform == None:
            image = T.ToTensor()(image)
            future_image = T.ToTensor()(future_image)
            future_image = T.Resize((32, 32))(future_image)
        elif self.transform is not None:     
            image = self.transform(image)
            future_image = self.transform(future_image)

        return image, (eeTarget, eePos, eeOri, future_image)

    def get_valid_imagePair(self, index: int) -> Tuple[str, str]:
        filename = self.df["imLoc"][index]
        no_error = False
        while no_error is False:
            try:
                future_filename = self.df["imLoc"][index + self.future_delta_target]
                no_error = True
            except KeyError:
                self.future_delta_target -= 1
        
        while self.check_sameDemo(filename, future_filename) == False:
            self.future_delta_target -= 1
            # print(f"near the end {self.future_delta_target}")
            if self.future_delta_target == 0:
                self.df = self.df.drop(index)
                self.df.reset_index(inplace=True, drop=True)
                # print(self.df)
                self.future_delta_target = self.delta_steps
                filename = self.df["imLoc"][index]
                print("removed row")
            future_filename = self.df["imLoc"][index + self.future_delta_target]
        return filename, future_filename

    def check_sameDemo(self, current_filename: str, future_filename: str) -> bool:
        current_demo = get_numEpisodeRun(current_filename)
        return True if get_numEpisodeRun(future_filename) == current_demo else False

    def filter_stopData(self):
        self.df = self.df.drop(self.df[self.df.loc[:,"stop"] == 1].index)
        self.df.reset_index(inplace=True, drop=True)

    @staticmethod
    def remove_unusable_data(df: pd.DataFrame):
        filter_func = lambda row: get_stepNum(row["imLoc"]) == 0 
        indices = df.apply(filter_func, axis=1)
        indices = pd.concat((indices, pd.Series(True)), ignore_index=True)
        indices.index -= 1
        indices = indices[1:]
        df = df.drop(indices[indices == True].index)
        df.reset_index(inplace=True)
        # print(df["index"])
        return df["index"]

    def calculate_statistics(self):
        dataloader = DataLoader(self, batch_size=120, num_workers=1)
        for (x, labels) in dataloader:
            try:
                input_image = torch.cat((input_image, x), axis=0)
                mi_image = torch.cat((mi_image, labels[-1]), axis=0)
                data = torch.cat((data, torch.cat(labels[:-1], dim=1)), axis=0)
            except NameError:
                input_image = x
                mi_image = labels[-1]
                data = torch.cat(labels[:-1], dim=1)
        input_std, input_mean = torch.std_mean(input_image, axis=[0,2,3])
        input_std = torch.ones(input_std.shape)
        input_mean = torch.zeros(input_mean.shape)
        print(f"input -->\tmean:{input_mean}, \tstd: {input_std}")
        mi_std, mi_mean = torch.std_mean(mi_image, axis=[0,2,3])
        # mi_std = torch.ones(mi_std.shape)
        # mi_mean = torch.zeros(mi_mean.shape)
        print(f"mi -->   \tmean:{mi_mean}, \tstd: {mi_std}")
        data_std, data_mean = torch.std_mean(data, axis=0)
        data_std = torch.ones(data_std.shape)
        data_mean = torch.zeros(data_mean.shape)
        print(f"data -->\tmean:{data_mean}, \tstd: {data_std}\n\n")
        return (input_std, input_mean), (mi_std, mi_mean), (data_std, data_mean)

    def get_transforms(self, get_stats: bool=False):
        def vector_transform(x: torch.Tensor):
            return (x - stats[2][1]) / stats[2][0]
        print("Calculating mean and standard deviation for data")
        stats = self.calculate_statistics()
        input_transform = T.Normalize(stats[0][1], stats[0][0])
        mi_transform = T.Normalize(stats[1][1], stats[1][0])
        # data_transform = T.Normalize(stats[2][1], stats[2][0])
        if get_stats == False:
            return input_transform, mi_transform, vector_transform
        else:
            return (input_transform, mi_transform, vector_transform), stats


    # def calculate_mean_and_std(self, img, mi, data):
    #     # def getImage(row):
    #     #     filename = row["imLoc"]
    #     #     image = Image.open(self.dataset_path + filename)
    #     #     return T.ToTensor(image)

    #     img = T.ToTensor(img)
    #     mi = T.ToTensor(mi)
        

        

