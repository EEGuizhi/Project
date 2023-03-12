import numpy as np
import torch
import torch.nn as nn

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
print(256/16)
print(256//16)
