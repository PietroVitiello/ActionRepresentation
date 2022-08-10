from abc import abstractclassmethod
from typing import List, Tuple

from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from .TL_eeVel import TL_eeVel
from .TL_aux import TL_aux
from .TL_stop import TL_stop
from .TL_onlyStop import TL_onlyStop
from .TL_MI import TL_motionImage

class SimDataset(Dataset):
    def __init__(
        self, 
        dataset_path,
        transform = None,
        filter_stop: bool=False,
        considered_indices: np.ndarray = None
    ) -> None:

        self.df = pd.read_csv(dataset_path + "data.csv")
        self.transform = transform
        self.dataset_path = dataset_path
        if filter_stop:
            self.filter_stopData()
        if considered_indices is not None:
            self.filter_indices(considered_indices)

    def filter_indices(self, indices):
        filter_func = lambda row: row["demo_id"] in indices
        filter_ids = self.df.apply(filter_func, axis=1)
        self.df: pd.DataFrame = self.df.drop(filter_ids[filter_ids == False].index)
        self.df.reset_index(inplace=True)

    @abstractclassmethod
    def __len__(self):
        pass

    @abstractclassmethod
    def __getitem__(self, index):
        pass

    @abstractclassmethod
    def filter_stopData(self):
        pass

    @staticmethod
    def get_train_val_ids(dataset: pd.DataFrame, split: float, n_demos: int= None):
        len_df = len(dataset)
        number_demos = dataset.iloc[len_df]["demo_id"] + 1
        if n_demos is not None:
            assert(n_demos <= number_demos)
            number_demos = n_demos
        training_len = int(np.round(number_demos * split))
        indices = np.arange(number_demos+1)
        indices = np.random.shuffle(indices)
        return indices[:training_len], indices[training_len:]

    @staticmethod
    def get_with_val( 
        dataset_path,
        train_val_split: float,
        transform = None,
        dataset_mode: str="eeVel",
        filter_stop: bool=False,
        n_demos: int=None
    ) -> Tuple[Dataset]:

        df = pd.read_csv(dataset_path + "data.csv")
        train_ids, val_ids = SimDataset.get_train_val_ids(df, train_val_split, n_demos)
        train_dataset = SimDataset.get(
            dataset_path,
            transform,
            dataset_mode,
            filter_stop,
            train_ids
        )
        val_dataset = SimDataset.get(
            dataset_path,
            transform,
            dataset_mode,
            filter_stop,
            val_ids
        )
        return train_dataset, val_dataset

    @staticmethod
    def get( 
        dataset_path,
        transform = None,
        dataset_mode: str="eeVel",
        filter_stop: bool=False,
        considered_indices: np.ndarray = None
    ) -> Dataset:

        if dataset_mode == "eeVel":
            return TL_eeVel(
                dataset_path,
                transform,
                filter_stop,
                considered_indices
            )
        elif dataset_mode == "aux":
            return TL_aux(
                dataset_path,
                transform,
                filter_stop
            )
        elif dataset_mode == "stop":
            return TL_stop(
                dataset_path,
                transform,
                filter_stop
            )
        elif dataset_mode == "onlyStop":
            return TL_onlyStop(
                dataset_path,
                transform,
                filter_stop
            )
        elif dataset_mode == "motionImage":
            return TL_motionImage(
                dataset_path,
                transform,
                filter_stop,
                considered_indices
            )
        else:
            raise Exception("The selected dataset mode is not supported")

            