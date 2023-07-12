# NCHUEE大學專題  組員: 陳柏翔 陳沛昀
import os
import torch
import numpy as np
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from hrnet import HighResolutionNet
from ocr import SpatialOCR_Module, SpatialGather_Module


class ConvBlocks(nn.Module):
    def __init__(self, in_ch, channels, kernel_sizes=None, strides=None, dilations=None, paddings=None, BatchNorm=nn.BatchNorm2d):
        super(ConvBlocks, self).__init__()
        self.num = len(channels)
        if kernel_sizes is None: kernel_sizes = [3 for c in channels]
        if strides is None: strides = [1 for c in channels]
        if dilations is None: dilations = [1 for c in channels]
        if paddings is None:
            paddings = [
                ((kernel_sizes[i] // 2) if dilations[i] == 1 else (kernel_sizes[i] // 2 * dilations[i])) for i in range(self.num)
            ]
        convs_tmp = []
        for i in range(self.num):
            if channels[i] == 1:
                convs_tmp.append(
                    nn.Conv2d(
                        in_ch if i == 0 else channels[i - 1], channels[i], kernel_size=kernel_sizes[i],
                        stride=strides[i], padding=paddings[i], dilation=dilations[i]
                    )
                )
            else:
                convs_tmp.append(nn.Sequential(
                    nn.Conv2d(
                        in_ch if i == 0 else channels[i - 1], channels[i], kernel_size=kernel_sizes[i],
                        stride=strides[i], padding=paddings[i], dilation=dilations[i], bias=False
                    ),
                    BatchNorm(channels[i]),
                    nn.ReLU()
                ))
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
    def __init__(self, in_ch, out_ch, num_classes, SE_maxpool=True, SE_softmax=False):
        super(IGGNet, self).__init__()
        self.hintEncoder = ConvBlocks(in_ch+num_classes, [256, 256, 256], [3, 3, 3], [2, 1, 1])
        self.conv1 = nn.Conv2d(256, 16, kernel_size=(1, 1))  # SE_Block
        self.conv2 = nn.Conv2d(16, out_ch, kernel_size=(1, 1))
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
        if Fc is not None:
            return f * Fc
        else:
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
    def __init__(self, cfg, pretrained_model_path=None):  # 假設傳入的config會是config.MODEL
        super(IKEM, self).__init__()
        self.image_size = cfg.IMAGE_SIZE
        ocr_width = 128

        # Hint Fusion Layer
        self.hint_fusion_layer = HintFusionLayer(cfg.IMAGE_SIZE[0], cfg.NUM_KEYPOINTS*2)

        # High Resolution Network
        self.hrnet = HighResolutionNet(width=32, num_classes=cfg.NUM_KEYPOINTS, ocr_width=128, small=False)

        # Interaction-Guided Gating Network
        last_inp_channels = self.hrnet.last_inp_channels
        self.iggnet = IGGNet(in_ch=256, out_ch=last_inp_channels, num_classes=cfg.NUM_KEYPOINTS, SE_maxpool=True, SE_softmax=False)

        # Object-Contextual Representations
        ocr_mid_channels = 2 * ocr_width
        ocr_key_channels = ocr_width
        self.conv3x3_ocr = nn.Sequential(
                nn.Conv2d(last_inp_channels, ocr_mid_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(ocr_mid_channels),
                nn.ReLU(inplace=True),
        )
        self.ocr_gather_head = SpatialGather_Module(cfg.NUM_KEYPOINTS)

        self.ocr_distri_head = SpatialOCR_Module(
            in_channels=ocr_mid_channels,
            key_channels=ocr_key_channels,
            out_channels=ocr_mid_channels,
            scale=1,
            dropout=0.05,
            norm_layer=nn.BatchNorm2d,
            align_corners=True
        )
        self.cls_head = nn.Conv2d(
            ocr_mid_channels, cfg.NUM_KEYPOINTS, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.aux_head = nn.Sequential(
            nn.Conv2d(last_inp_channels, last_inp_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(last_inp_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_inp_channels, cfg.NUM_KEYPOINTS, kernel_size=1, stride=1, padding=0, bias=True)
        )

    # Load pretrained model param
        if pretrained_model_path is not None:
            model_dict = self.state_dict()
            if not os.path.exists(pretrained_model_path):
                print("Error: Pretrained model file path does not exist.")
                exit(1)
            pretrained_dict = torch.load(pretrained_model_path)
            for key in list(pretrained_dict.keys()):
                if key[0:5] == "conv1": pretrained_dict[key.replace("conv1", "hint_fusion_layer.image_encoder.0")] = pretrained_dict.pop(key)
                elif key[0:5] == "conv2": pretrained_dict[key.replace("conv2", "hint_fusion_layer.last_encoder.0")] = pretrained_dict.pop(key)
                elif key[0:3] == "bn1": pretrained_dict[key.replace("bn1", "hint_fusion_layer.image_encoder.1")] = pretrained_dict.pop(key)
                elif key[0:3] == "bn2": pretrained_dict[key.replace("bn2", "hint_fusion_layer.last_encoder.1")] = pretrained_dict.pop(key)
                else: pretrained_dict["hrnet."+key] = pretrained_dict.pop(key)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)

    def forward(self, hint_heatmap:Tensor, prev_heatmap:Tensor, input_image:Tensor):
        feature_map = self.hint_fusion_layer(input_image, hint_heatmap, prev_heatmap)
        Fh, Fc = self.hrnet(feature_map)  # Fh以及Fc
        feature_map = self.iggnet(hint_heatmap, Fh, Fc)

        out_aux = self.aux_head(feature_map)  # aux_head : conv norm relu conv (soft object regions), output channel: num_classes
        feature_map = self.conv3x3_ocr(feature_map)  # conv3x3_ocr : conv norm relu (pixel representation

        context = self.ocr_gather_head(feature_map, out_aux)  # context :  batch x c x num_keypoint x 1, feature_map: batch, c, H, W
        feature_map = self.ocr_distri_head(feature_map, context)
        out = self.cls_head(feature_map)

        pred_logit = F.interpolate(pred_logit, size=self.image_size, mode='bilinear', align_corners=True)
        aux_pred_logit = F.interpolate(aux_pred_logit, size=self.image_size, mode='bilinear', align_corners=True)

        return pred_logit, aux_pred_logit
