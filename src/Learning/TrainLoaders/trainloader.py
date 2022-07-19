from torch.utils.data import Dataset
from .TL_eeVel import TL_eeVel
from .TL_aux import TL_aux
from .TL_stop import TL_stop
from .TL_onlyStop import TL_onlyStop
from .TL_MI import TL_motionImage

class SimDataset(Dataset):
    def __init__(self) -> None:
        pass

    def get(
        self, 
        dataset_path,
        transform = None,
        dataset_mode: str="eeVel",
        filter_stop: bool=False
    ) -> Dataset:

        if dataset_mode == "eeVel":
            return TL_eeVel(
                dataset_path,
                transform,
                filter_stop
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
                filter_stop
            )
        else:
            raise Exception("The selected dataset mode is not supported")