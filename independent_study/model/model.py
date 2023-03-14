# NCHU EE
import torch
import numpy
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from hrnet import HighResolutionNet

# class ScaleLayer(nn.Module):  # from https://github.com/seharanul17/interactive_keypoint_estimation/blob/8a6f28df6da5728dcf99827333f7f7620fda28b8/model/iterativeRefinementModels/RITM_SE_HRNet32.py#L132
#     def __init__(self, init_value=1.0, lr_mult=1):
#         super().__init__()
#         self.lr_mult = lr_mult
#         self.scale = nn.Parameter(
#             torch.full((1,), init_value / lr_mult, dtype=torch.float32)
#         )

#     def forward(self, x):
#         scale = torch.abs(self.scale * self.lr_mult)
#         return x * scale

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernal, stride, padding=None, dilation=None, bn=nn.BatchNorm2d):
        super(ConvBlock, self).__init__()
        

class IGGNet(nn.Module):  # Interaction-Guided Gating Network
    def __init__(self, ):
        super(IGGNet).__init__()
        
    def forward(self, hint_heatmap: Tensor, feature_map: Tensor, cfg: dict):
        pass


class HintFusionLayer(nn.Module):
    def __init__(self, im_ch:int, in_ch:int, out_ch=64):
        """
        im_ch: Num of input image's channels
        in_ch: Num of "hintmap_encoder's input channels", equals to keypoints*2
        out_ch: Num of channels equals to "input channels of HRNet"
        """
        super(HintFusionLayer, self).__init__()
        init_value = 0.05
        lr_mult = 1
        self.scale = nn.Parameter(
            torch.full((1,), init_value / lr_mult, dtype=torch.float32)
        )
        self.hintmap_encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=16, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=16, out_channels=out_ch, kernel_size=3, stride=2, padding=1)
        )
        self.image_encoder = nn.Sequential(
            nn.Conv2d(in_channels=im_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.last_encoder = nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, input_image, hint_heatmap, prev_heatmap):
        f = torch.cat((hint_heatmap, prev_heatmap))
        f = self.hintmap_encoder(f)
        scale = torch.abs(self.scale * self.lr_mult)
        f = f * scale
        f = f + self.image_encoder(input_image)
        f = self.last_encoder(f)
        return f        


class IKEM(nn.Module):  # Interaction Keypoint Estimation Model
    def __init__(self, cfg):  # 假設傳入的config會是config.MODEL
        super(IKEM, self).__init__()
        
        # Hint Fusion Layer
        self.hint_fusion_layer = HintFusionLayer(cfg.IMAGE_CHANNEL, cfg.NUM_KEYPOINTS*2, cfg.HintFusionLayer.ENCODE_CHANNEL)
        
        # 引入 HRNet
        self.hrnet = HighResolutionNet()
        
        # 引入 Interaction-Guided Gating Network
        self.ignet = IGGNet()
        
    def forward(self, x):
        pass
