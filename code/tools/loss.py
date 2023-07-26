import torch
import torch.nn as nn
import numpy as np

class CustomLoss(nn.Module):
    def __init__(self, use_coord_loss, use_morph_loss):
        super(CustomLoss, self).__init__()
        self.use_coord_loss = use_coord_loss
        self.use_morph_loss = use_morph_loss
        self.bce_loss = nn.BCELoss()
        self.mae_criterion = nn.L1Loss()
        self.angle_criterion = nn.CosineEmbeddingLoss()

    def forward(self, pred_coord, pred_heatmap, label_coord, label_heatmap):
        loss, pred_heatmap = self.get_heatmap_loss(pred_heatmap=pred_heatmap, label_heatmap=label_heatmap)
        if self.use_morph_loss:
            loss += self.get_morph_loss(pred_coord, label_coord, alpha=0.01)
        if self.use_coord_loss:
            coord_loss = self.mae_criterion(pred_coord, label_coord)
            loss += 0.01 * coord_loss
        return loss

    def get_heatmap_loss(self, pred_heatmap:torch.Tensor, label_heatmap:torch.Tensor):
        pred_heatmap = pred_heatmap.sigmoid()
        heatmap_loss = self.bce_loss(pred_heatmap, label_heatmap)
        return heatmap_loss, pred_heatmap

    def get_morph_loss(self, pred_coords:torch.Tensor, label_coords:torch.Tensor, alpha:float):
        # Dim of inputs = (batch, 68, 2)
        # Random indexes
        rand_index = np.random.choice(4, size=pred_coords.shape[1]//4).tolist()
        for i in range(pred_coords.shape[1]//4):
            rand_index[i] += i*4

        # Vectors
        pred_vecA = pred_coords[:, rand_index[2:], :] - pred_coords[:, rand_index[1:-1], :]
        pred_vecB = pred_coords[:, rand_index[:-2], :] - pred_coords[:, rand_index[1:-1], :]
        label_vecA = label_coords[:, rand_index[2:], :] - label_coords[:, rand_index[1:-1], :]
        label_vecB = label_coords[:, rand_index[:-2], :] - label_coords[:, rand_index[1:-1], :]

        # Distance
        pred_vecA_dis = torch.norm(pred_vecA, dim=-1)
        label_vecA_dis = torch.norm(label_vecA, dim=-1)

        # Angles between vecA & vecB
        pred_angle = torch.atan2(pred_vecA[:, :, 0], pred_vecA[:, :, 1]) - torch.atan2(pred_vecB[:, :, 0], pred_vecB[:, :, 1])
        label_angle = torch.atan2(label_vecA[:, :, 0], label_vecA[:, :, 1]) - torch.atan2(label_vecB[:, :, 0], label_vecB[:, :, 1])

        # Vec (sin, cos) of angles
        pred_angle_vec = torch.stack((torch.cos(pred_angle), torch.sin(pred_angle)), -1)
        label_angle_vec = torch.stack((torch.cos(label_angle), torch.sin(label_angle)), -1)

        # Cosine loss
        N = pred_angle.shape[0] * pred_angle.shape[1]
        label_similarity = torch.ones(N, dtype=pred_angle_vec.dtype, device=pred_coords.device)
        morph_loss = self.angle_criterion(pred_angle_vec.reshape(N, 2), label_angle_vec.reshape(N, 2), label_similarity)

        # Distance loss
        morph_loss += self.mae_criterion(pred_vecA_dis, label_vecA_dis)

        return alpha * morph_loss
