# NCHUEE大學專題  組員: 陳柏翔 陳沛昀
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


class ConvBlocks(nn.Module):
    def __init__(self, in_ch, channels, kernel_sizes=None, strides=None, dilations=None, paddings=None,
                 BatchNorm=nn.BatchNorm2d):
        super(ConvBlocks, self).__init__()
        self.num = len(channels)
        if kernel_sizes is None: kernel_sizes = [3 for c in channels]
        if strides is None: strides = [1 for c in channels]
        if dilations is None: dilations = [1 for c in channels]
        if paddings is None: paddings = [
            ((kernel_sizes[i] // 2) if dilations[i] == 1 else (kernel_sizes[i] // 2 * dilations[i])) for i in
            range(self.num)]
        convs_tmp = []
        for i in range(self.num):
            if channels[i] == 1:
                convs_tmp.append(nn.Conv2d(
                    in_ch if i == 0 else channels[i - 1], channels[i], kernel_size=kernel_sizes[i],
                    stride=strides[i], padding=paddings[i], dilation=dilations[i])
                )
            else:
                convs_tmp.append(nn.Sequential(
                    nn.Conv2d(in_ch if i == 0 else channels[i - 1], channels[i], kernel_size=kernel_sizes[i],
                              stride=strides[i], padding=paddings[i], dilation=dilations[i], bias=False),
                    BatchNorm(channels[i]), nn.ReLU())
                )
        self.convs = nn.Sequential(*convs_tmp)

        # weight initialization
        for m in self.convs.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.convs(x)

class IGGNet(nn.Module):  # Interaction-Guided Gating Network (Fc另外寫)
    def __init__(self, in_ch, mid_ch1, out_ch, num_classes, SE_maxpool=False, SE_softmax=False, input_channel=256):
        super(IGGNet, self).__init__()
        self.hintEncoder = ConvBlocks(input_channel+num_classes, [256, 256, 256], [3, 3, 3], [2, 1, 1])
        self.conv1 = nn.Conv2d(in_ch, mid_ch1, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(mid_ch1, out_ch, kernel_size=(1, 1))
        self.SE_maxpool = SE_maxpool
        self.SE_softmax = SE_softmax

    def forward(self, hint_heatmap:Tensor, Fh:Tensor, Fc:Tensor):
        hint = F.interpolate(input=hint_heatmap, size=Fh.size()[2:], mode="bilinear", align_corners=True)
        f = torch.cat((Fh, hint), dim=1)
        f = self.hintEncoder(f)
        if self.SE_maxpool:
            f = f.max(-1)[0].max(-1)[0]
            f = f[:, :, None, None]
        else:
            f = f.mean(-1, keepdim=True).mean(-2, keepdim=True)
        f = self.conv1(f).relu() 
        f = self.conv2(f)
        if self.SE_softmax:  # 得到架構中的A
            f = f.softmax(1)
        else:
            f = f.sigmoid()
        f = f * Fc
        return f    

class HintFusionLayer(nn.Module):
    def __init__(self, im_ch:int, in_ch:int, out_ch:int=64, ScaleLayer:bool=True):
        """
        im_ch: Num of input image's channels
        in_ch: Num of "hintmap_encoder's input channels", equals to keypoints*2
        out_ch: Num of channels equals to "input channels of HRNet"
        """
        super(HintFusionLayer, self).__init__()
        self.ScaleLayer = ScaleLayer
        init_value = 0.05
        lr_mult = 1
        self.hintmap_encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=16, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=16, out_channels=out_ch, kernel_size=3, stride=2, padding=1)
        )
        self.lr_mult = lr_mult
        self.scale = nn.Parameter(
            torch.full((1,), init_value / lr_mult, dtype=torch.float32)
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

    def forward(self, input_image:Tensor, hint_heatmap:Tensor, prev_heatmap:Tensor):
        f = torch.cat((hint_heatmap, prev_heatmap))
        f = self.hintmap_encoder(f)
        if self.ScaleLayer:
            scale = torch.abs(self.scale * self.lr_mult)
            f = f * scale
        f = f + self.image_encoder(input_image)
        f = self.last_encoder(f)
        return f


class IKEM(nn.Module):  # Interaction Keypoint Estimation Model
    def __init__(self, cfg):  # 假設傳入的config會是config.MODEL
        super(IKEM, self).__init__()

        # Hint Fusion Layer
        self.hint_fusion_layer = HintFusionLayer(cfg.IMAGE_SIZE[0], cfg.NUM_KEYPOINTS*2, cfg.HintFusionLayer.out_channel)

        # 引入 HRNet
        self.hrnet = HighResolutionNet()

        # 引入 Interaction-Guided Gating Network
        self.iggnet = IGGNet()
        
    def forward(self, hint_heatmap:Tensor, prev_heatmap:Tensor, input_image:Tensor):
        feature_map = self.hint_fusion_layer(input_image, hint_heatmap, prev_heatmap)
        downsampled_feature_map, intermediate_feature_map = self.hrnet(feature_map)  # Fh以及Fc
        out_heatmap = self.IGGNet(hint_heatmap, downsampled_feature_map, intermediate_feature_map)
        
        
