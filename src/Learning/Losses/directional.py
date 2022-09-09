import torch
import torch.nn as nn

class DirectionalLoss():

    def __init__(self) -> None:
        self.mse = nn.MSELoss()
        self.cosine_loss = nn.CosineSimilarity()

    def __call__(self, pred, label):
        eeVel_pred = pred[:,:3]
        eeVel_label = label[:,:3]
        dir_similarity = self.cosine_loss(eeVel_pred, eeVel_label)
        mse = self.mse(pred, label)
        return dir_similarity + mse