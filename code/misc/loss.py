import torch
import torch.nn as nn
import torch.nn.functional as F
from munch import Munch

class LossManager():
    def __init__(self, use_coord_loss, heatmap_maker):
        self.use_coord_loss = use_coord_loss
        self.heatmap_maker = heatmap_maker
        self.mae_criterion = nn.L1Loss()
        self.bce_loss = nn.BCELoss()

    def __call__(self, pred_heatmap, label):
        loss, pred_heatmap = self.get_heatmap_loss(pred_heatmap=pred_heatmap, label_heatmap=label.heatmap)
        pred_coord = self.heatmap_maker.get_heatmap2sargmax_coord(pred_heatmap=pred_heatmap)
        if self.use_coord_loss:
            coord_loss = self.mae_criterion(pred_coord, label.coord)
            loss += 0.01*coord_loss
        return loss

    def get_heatmap_loss(self, pred_heatmap:torch.Tensor, label_heatmap:torch.Tensor):
        pred_heatmap = pred_heatmap.sigmoid()
        heatmap_loss = self.bce_loss(pred_heatmap, label_heatmap)
        return heatmap_loss, pred_heatmap
