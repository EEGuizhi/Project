import numpy as np
import torch
import os

import torch.nn as nn
from torch import Tensor

### python index test ###
# heatmap = np.array([[0.2, 0.5, 0.8], [0.9, 0.3, 0.6], [0.1, 0.7, 0.4]])
# threshold = 0.5
# print("heatmap:", heatmap)  # output: heatmap: [[0.2 0.5 0.8]
#                             #                   [0.9 0.3 0.6]
#                             #                   [0.1 0.7 0.4]]
# heatmap[heatmap < threshold] = 0
# print("heatmap:", heatmap)  # output: heatmap: [[0.2 0.5 0.8]
#                             #                   [0.9 0.  0.6]
#                             #                   [0.  0.7 0. ]]


### BCELoss test ###
# m = nn.Sigmoid()
# # bce_loss = nn.BCELoss(reduction='sum')  # 沒有平均
# bce_loss = nn.BCELoss()  # 有平均
# output = torch.tensor([-1, -2, -3, 1, 2, 3], dtype=torch.float32)
# output = m(output)  # 先處理一次 (將數值轉到0~1之間)
# target = torch.tensor([0, 1, 0, 0, 0, 1], dtype=torch.float32)
# loss = bce_loss(output, target)
# print(loss)


### dataset 16 ###
# [[[2, 4], [6, 8], [3, 5], [18, 20], [10, 12], [7, 9], [14, 16], [11, 13], [22, 24], [23, 25], [19, 21], [26, 28], [15, 17], [30, 32], [4, 6], [8, 12], [8, 10], [31, 33], [0, 2], [27, 29], [34, 36], [6, 10], [38, 40], [4, 8], [1, 3], [12, 16], [35, 37], [12, 14], [46, 48], [29, 33], [29, 31], [2, 6], [33, 37], [39, 41], [42, 44], [17, 21], [27, 31], [35, 39], [23, 27], [5, 7], [10, 14], [9, 13], [16, 20], [25, 27], [26, 30], [1, 5], [31, 35], [43, 45], [21, 25], [19, 23], [6, 12], [16, 18], [5, 9], [33, 35], [14, 18], [28, 30], [25, 29], [0, 4], [28, 32], [55, 57], [59, 61], [9, 11], [21, 23], [24, 26], [13, 17], [36, 40], [18, 22], [47, 49], [7, 11], [3, 7]],
#  [[2, 63, 4], [2, 62, 4], [2, 61, 4], [2, 59, 4], [2, 60, 4], [2, 58, 4], [3, 63, 5], [3, 62, 5], [2, 57, 4], [2, 56, 4], [2, 55, 4], [3, 61, 5], [3, 60, 5], [2, 54, 4], [6, 63, 8], [3, 59, 5], [3, 58, 5], [2, 53, 4], [6, 62, 8], [2, 52, 4], [2, 51, 4], [6, 61, 8], [7, 62, 9], [3, 56, 5], [7, 63, 9], [3, 57, 5], [2, 50, 4], [6, 59, 8], [6, 60, 8], [3, 54, 5], [3, 55, 5], [7, 60, 9], [6, 58, 8], [2, 49, 4], [7, 61, 9], [2, 48, 4], [11, 62, 13], [11, 63, 13], [6, 57, 8], [0, 63, 2], [2, 47, 4], [7, 58, 9], [2, 46, 4], [3, 52, 5], [7, 59, 9], [10, 63, 12], [0, 62, 2], [3, 53, 5], [6, 56, 8], [6, 55, 8], [3, 50, 5], [0, 61, 2], [10, 62, 12], [11, 61, 13], [3, 51, 5], [7, 65, 9], [11, 60, 13], [6, 54, 8], [0, 60, 2], [7, 56, 9], [10, 61, 12], [2, 44, 4], [0, 59, 2], [2, 45, 4], [7, 57, 9], [0, 58, 2], [11, 59, 13], [11, 58, 13], [6, 53, 8], [3, 48, 5]]]


### 並聯 ###
# import torch
# import torch.nn as nn

# class ParallelConv(nn.Module):
#     def __init__(self, in_channels, out_channels, num_convs):
#         super(ParallelConv, self).__init__()
#         self.num_convs = num_convs
#         self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) for i in range(num_convs)])

#     def forward(self, x):
#         outputs = []
#         for i in range(self.num_convs):
#             outputs.append(self.convs[i](x))
#         return torch.cat(outputs, dim=1)


### dictionary test ###
# cfg = dict(num_modules=1, num_branches=1, block='BOTTLENECK', num_blocks=(4), num_channels=(64, ))
# cfg = dict(
#     MODEL = dict(
#         NAME = "cls_hrnet",
#         IMAGE_SIZE = (224, 224)
#     )
# )

# print(">> cfg :", cfg)
# print("cfg['MODEL'] :", cfg['MODEL'])
# print("cfg.MODEL :", cfg.MODEL)


### operation '//' test ###
# print(256/16)
# print(256//16)


### next() ###
# list=[1,2,3,4]
# it = iter(list)
# print(next(it))


### nn.Parameter test ###
# class ScaleLayer(nn.Module):
#     def __init__(self, init_value=1.0, lr_mult=1):
#         super().__init__()
#         self.lr_mult = lr_mult
#         self.scale = nn.Parameter(
#             torch.full((1,), init_value / lr_mult, dtype=torch.float32)
#         )
        
#         # a = torch.tensor([3, 4], dtype=torch.float64)

#     def forward(self, x):
#         scale = torch.abs(self.scale * self.lr_mult)
#         return x * scale


# model = ScaleLayer()

# model.state_dict()
# for para in model.parameters():
#     print(para)


### random select test ###
# test_paths = ['dataset/img_A.jpg','dataset/img_B.jpg','dataset/img_C.jpg','dataset/img_D.jpg','dataset/img_E.jpg','dataset/img_F.jpg','dataset/img_G.jpg']
# print("test_paths: {}".format(test_paths))

# idx = sorted(np.random.choice(range(len(test_paths)), size=3, replace=False, p=None))
# print(">> idx len: {}, idx: {}".format(len(idx), idx))

# # select_img_paths = np.array(test_paths)[idx].tolist()
# select_img_paths = np.array(test_paths)[idx]
# print(">> select_lmg_paths: {}".format(select_img_paths))


### load image test ###
# from PIL import Image
# image_path = "D:/python/interactive_keypoint_estimation/code/data/dataset16/boostnet_labeldata/data/test/sunhl-1th-01-Mar-2017-310 C AP.jpg"

# img = np.array(Image.open(os.path.join(image_path)))
# print(">> origin size: {}".format(img.shape))

# img = np.repeat(np.array(Image.open(os.path.join(image_path)))[:,:,None], 3, axis=-1)
# print(">> new size: {}".format(img.shape))


### loading .mat file test ###
# import scipy.io
# content = scipy.io.loadmat("D:/python/interactive_keypoint_estimation/code/data/dataset16/boostnet_labeldata/labels/test/sunhl-1th-01-Mar-2017-310 C AP.jpg.mat")
# print(">> sunhl-1th-01-Mar-2017-310 C AP.jpg.mat: \n{}".format(content))

# coords = content['p2']
# print(">> keypoints coord:\n{}".format(coords))
# print(">> keypoints coord[:, 1]:\n{}".format(coords[:, 1]))


### copy list test ###
# import copy
# print("===")
# a = [1, 2, 3, [99, 100]]
# b = a
# print("operation: b = a, b[3] = 4")
# print(">> a: {}, b: {}".format(a, b))
# b[3] = 4
# print(">> a: {}, b: {}".format(a, b))

# print("===")
# a = [1, 2, 3, [99, 100]]
# b = a.copy()  # or "b = list(a)", "b = a[:]"
# print("operation: b = a.copy(), b[3] = 4")
# print(">> a: {}, b: {}".format(a, b))
# b[3] = 4
# print(">> a: {}, b: {}".format(a, b))

# print("===")
# a = [1, 2, 3, [99, 100]]
# b = a.copy()  # or "b = list(a)", "b = a[:]" (= shallow copy)
# print("operation: b = a.copy(), b[3][0] = 4")
# print(">> a: {}, b: {}".format(a, b))
# b[3][0] = 4
# print(">> a: {}, b: {}".format(a, b))

# print("===")
# a = [1, 2, 3, [99, 100]]
# b = copy.deepcopy(a)  # deep copy
# print("operation: b = copy.deepcopy(a), b[3][0] = 4")
# print(">> a: {}, b: {}".format(a, b))
# b[3][0] = 4
# print(">> a: {}, b: {}".format(a, b))

table = [{'key1': 1, 'key2': 2}, {'key1': 3, 'key2': 4}, {'key1': 5, 'key2': 6}]
for _, item in enumerate(table):
    item['key1'] = 99
print(">> table: {}".format(table))
