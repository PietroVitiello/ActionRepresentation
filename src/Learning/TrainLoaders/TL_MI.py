from typing import Tuple
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image

from ..utils.motion_image import get_motionImage
from ..utils.utils_dataloader import get_numEpisodeRun, get_stepNum

class TL_motionImage(Dataset):
    def __init__(
        self, 
        dataset_path,
        transform = None,
        filter_stop: bool = False,
        delta_steps: int = 1
    ) -> None:

        self.df = pd.read_csv(dataset_path + "data.csv")
        if filter_stop:
            self.filter_stopData()
        self.cleaned_df = self.remove_unusable_data(self.df)

        self.transform = transform
        self.dataset_path = dataset_path
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
        motion_image = get_motionImage(image, future_image, resized_side=32)

        if self.transform is not None:
            image = self.transform(image)
            motion_image = self.transform(motion_image)

        return image, (eeTarget, eePos, eeOri, cPos, motion_image)

    def get_valid_imagePair(self, index: int) -> Tuple[str, str]:
        filename = self.df["imLoc"][index]
        try:
            future_filename = self.df["imLoc"][index + self.future_delta_target]
        except KeyError:
            self.future_delta_target -= 1
            future_filename = self.df["imLoc"][index + self.future_delta_target]
        
        while self.check_sameDemo(filename, future_filename) == False:
            self.future_delta_target -= 1
            print(f"near the end {self.future_delta_target}")
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

