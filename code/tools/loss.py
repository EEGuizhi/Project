import torch
import torch.nn as nn
from .heatmap_maker import HeatmapMaker

class CustomLoss(nn.Module):
    def __init__(self, use_coord_loss, use_angle_loss, heatmap_maker=HeatmapMaker):
        super(CustomLoss, self).__init__()
        self.use_coord_loss = use_coord_loss
        self.use_angle_loss = use_angle_loss
        self.heatmap_maker = heatmap_maker
        self.bce_loss = nn.BCELoss()
        self.mae_criterion = nn.L1Loss()

    def forward(self, pred_heatmap, label, label_heatmap):
        loss, pred_heatmap = self.get_heatmap_loss(pred_heatmap=pred_heatmap, label_heatmap=label_heatmap)
        pred_coord = self.heatmap_maker.heatmap2sargmax_coord(pred_heatmap=pred_heatmap)
        if self.use_angle_loss:
            loss += self.get_angle_loss(pred_coord, label)
        if self.use_coord_loss:
            coord_loss = self.mae_criterion(pred_coord, label)
            loss += 0.01*coord_loss
        return loss

    def get_heatmap_loss(self, pred_heatmap:torch.Tensor, label_heatmap:torch.Tensor):
        pred_heatmap = pred_heatmap.sigmoid()
        heatmap_loss = self.bce_loss(pred_heatmap, label_heatmap)
        return heatmap_loss, pred_heatmap

    def get_angle_loss(self, pred_coords:torch.Tensor, label_coords:torch.Tensor):
        # Dim of inputs = (batch, 68, 2)
        for s in pred_coords.shape[0]:
            pred = torch.stack([
                pred_coords[s, [i*4   for i in range(pred_coords.shape[1]//4)], :],
                pred_coords[s, [i*4+1 for i in range(pred_coords.shape[1]//4)], :],
                pred_coords[s, [i*4+2 for i in range(pred_coords.shape[1]//4)], :],
                pred_coords[s, [i*4+3 for i in range(pred_coords.shape[1]//4)], :]
            ])
            label = torch.stack([
                label_coords[s, [i*4   for i in range(label_coords.shape[1]//4)], :],
                label_coords[s, [i*4+1 for i in range(label_coords.shape[1]//4)], :],
                label_coords[s, [i*4+2 for i in range(label_coords.shape[1]//4)], :],
                label_coords[s, [i*4+1 for i in range(label_coords.shape[1]//4)], :]
            ])
            pred_vecA = pred[2:pred.shape[0], :] - pred[1:pred.shape[0]-1, :]
            pred_vecB = pred[0:pred.shape[0]-2, :] - pred[1:pred.shape[0]-1, :]
            pred_angle = torch.atan(pred_vecA[:, 0] / pred_vecA[:, 1]) - torch.atan(pred_vecB[:, 0] / pred_vecB[:, 1])
            label_vecA = label[2:label.shape[0], :] - label[1:label.shape[0]-1, :]
            label_vecB = label[0:label.shape[0]-2, :] - label[1:label.shape[0]-1, :]
            label_angle = torch.atan(label_vecA[:, 0] / label_vecA[:, 1]) - torch.atan(label_vecB[:, 0] / label_vecB[:, 1])
            return self.mae_criterion(pred_angle, label_angle)

def find_worst_index(pred_coords:torch.Tensor, label_coords:torch.Tensor):
    # Dim of inputs = (68, 2)
    diff_coords = torch.pow(pred_coords - label_coords, 2)
    diff_coords = torch.sum(diff_coords, dim=-1)
    return torch.argmax(diff_coords).item()
