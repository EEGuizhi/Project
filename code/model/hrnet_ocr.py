# NCHUEE大學專題  組員: 陳柏翔 陳沛昀
import os
import torch
import numpy as np
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from .hrnet import HighResolutionNet
from .ocr import SpatialOCR_Module, SpatialGather_Module



class hrnet_ocr(nn.Module):  # HRNet OCR
    def __init__(self, image_size=(512, 256), im_ch:int=3, out_ch:int=64, num_of_keypoints=68, pretrained_model_path=None):  # 假設傳入的config會是config.MODEL
        super(hrnet_ocr, self).__init__()
        self.image_size = image_size
        ocr_width = 128

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

        # High Resolution Network
        self.hrnet = HighResolutionNet(width=32, ocr_width=128, small=False)

        # Object-Contextual Representations
        ocr_mid_channels = 2 * ocr_width
        ocr_key_channels = ocr_width
        self.conv3x3_ocr = nn.Sequential(
                nn.Conv2d(out_ch, ocr_mid_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(ocr_mid_channels),
                nn.ReLU(inplace=True),
        )
        self.ocr_gather_head = SpatialGather_Module(num_of_keypoints)

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
            ocr_mid_channels, num_of_keypoints, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.aux_head = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, num_of_keypoints, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # Load pretrained model param
        if pretrained_model_path is not None:
            model_dict = self.state_dict()
            if not os.path.exists(pretrained_model_path):
                print("Error: Pretrained model file path does not exist.")
                exit(1)
            print("Loading Pretrained model..")
            pretrained_dict = torch.load(pretrained_model_path)
            for key in list(pretrained_dict.keys()):
                if key[0:10] == "last_layer": pretrained_dict["hrnet."+key.replace("last_layer", "aux_head")] = pretrained_dict.pop(key)
                else: pretrained_dict["hrnet."+key] = pretrained_dict.pop(key)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)

    def forward(self, input_image:Tensor):
        feature_map = self.image_encoder(input_image)
        feature_map = self.last_encoder(feature_map)
        Fh, Fc = self.hrnet(feature_map)  # Fh以及Fc
        feature_map = Fc

        aux_out = self.aux_head(feature_map)  # aux_head : conv norm relu conv (soft object regions), output channel: num_classes
        feature_map = self.conv3x3_ocr(feature_map)  # conv3x3_ocr : conv norm relu (pixel representation

        context = self.ocr_gather_head(feature_map, aux_out)  # context :  batch x c x num_keypoint x 1, feature_map: batch, c, H, W
        feature_map = self.ocr_distri_head(feature_map, context)
        out = self.cls_head(feature_map)

        pred_logit = F.interpolate(out, size=self.image_size, mode='bilinear', align_corners=True)
        aux_pred_logit = F.interpolate(aux_out, size=self.image_size, mode='bilinear', align_corners=True)

        return pred_logit, aux_pred_logit
