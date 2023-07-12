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
# print(heatmap[:2])


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

# table = [{'key1': 1, 'key2': 2}, {'key1': 3, 'key2': 4}, {'key1': 5, 'key2': 6}]
# for _, item in enumerate(table):
#     item['key1'] = 99
# print(">> table: {}".format(table))


### Array test ###
# from tqdm.auto import tqdm
# import time

# test = torch.tensor([1, 2] + [77, 88])
# print(">> test: {}".format(test))
# print(">> test.sigmoid:",test.sigmoid())

# NUM_KEYPOINTS = 68
# num_dist = [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256, 1 / 512, 1 / 1024, 1 / 2048, 1 / 4096, 1 / 4096]+[0 for _ in range(NUM_KEYPOINTS-13)]
# num_hint = np.random.choice(range(NUM_KEYPOINTS), size=None, p=num_dist)  # 產生次數
# hint_indices = np.random.choice(range(NUM_KEYPOINTS), size=num_hint, replace=False) #[1,2,3]
# print(">> num_hint:\n{}\n>> hint_indices:\n{}".format(num_hint, hint_indices))

# for i, data in enumerate(tqdm(num_dist)):
#     print("\n>> i: {}, data: {}".format(i, data))
#     time.sleep(0.1)

# batch_metric_value = Tensor([
#     # batch 1
#     [0.11, 0.12, 0.13],  # pic 1, 3 keypoints
#     [0.21, 0.22, 0.23],  # pic 2, 3 keypoints
#     [0.31, 0.32, 0.33] # pic 3, 3 keypoints
# ])

# batch_hint_index = [[1, 3], [2], []]

# with torch.no_grad():
#     for j, idx in enumerate(batch_hint_index):
#         if idx is not None:
#             print(">> idx:", idx)
#             batch_metric_value[j, idx] = torch.full_like(batch_metric_value[j, idx], -1000)
#     worst_index = batch_metric_value.argmax(-1, keepdim=True)
# print(">> worst_index:\n", worst_index)

# a = np.array([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]])
# b = np.array([0, 2])
# print(">>", a[0, b])



###
# a = Tensor([1, 2, 3])
# a = torch.stack([a, a, a])
# print(a)

# a = torch.tensor([[1]])
# a = torch.cat((a, a), 1)
# print(a)

# import cv2
# import math
# import numpy as np
# from torchvision import transforms
# from torchvision.io.image import ImageReadMode
# from torchvision.io import read_image

# train_transform = transforms.Compose([
#      transforms.ToPILImage(),
#      transforms.Resize((512, 256)),
#      transforms.RandomHorizontalFlip(p=0.5),
#      transforms.ToTensor()
# ])

# a = read_image("Testing\\test_detect_output.png", mode=ImageReadMode.GRAY)
# print(a.shape)

# b = train_transform(a).numpy()
# print(b.shape)
# b *= 255
# b.astype(np.int8)
# b = np.transpose(b, (1, 2, 0))
# cv2.imwrite("test.jpg", b)

# prob = np.array([math.pow(2, -i) for i in range(5)])
# prob = prob.tolist() / prob.sum()
# print(prob)

# import torch
# import torch.nn as nn

# class CustomLoss(nn.Module):
#     def __init__(self):
#         super(CustomLoss, self).__init__()
#         self.base_loss_fn = nn.MSELoss()

#     def forward(self, output, target):
#         base_loss = self.base_loss_fn(output, target)
#         print(">> Output in CustomLoss:", output)
#         max_diff = torch.max(output) - torch.max(target)
#         final_loss = base_loss + max_diff
#         return final_loss

# # 假设模型为一个简单的全连接网络
# model = nn.Linear(in_features=5, out_features=1)
# loss_fn = CustomLoss()

# # 生成一个随机的批次数据
# batch_size = 8
# input_data = torch.randn(batch_size, 5)
# target_data = torch.randn(batch_size, 1)

# # 将输入数据传递给模型，获取模型的预测输出
# output = model(input_data)

# # 计算损失函数的值
# loss = loss_fn(output, target_data)

# # 打印损失值
# print(input_data)
# print(output)
# print(target_data)
# print("Batch Loss:", loss.item())

# a = np.array([1, 2, 3])
# print(a)
# print(list(a))
# print([a, a, a])

# a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# remove_idx = [1, 3, 5]
# for i, idx in enumerate(remove_idx):
#     del a[idx-i]
# print(a)


### nn.Parameter test ###
class ScaleLayer(nn.Module):
    def __init__(self, init_value=1.0, lr_mult=1):
        super(ScaleLayer, self).__init__()
        self.lr_mult = lr_mult
        self.scale = nn.Parameter(
            torch.full((1,), init_value / lr_mult, dtype=torch.float32)
        )

    def forward(self, x):
        scale = torch.abs(self.scale * self.lr_mult)
        return x * scale


class TopModel(nn.Module):
    def __init__(self, im_ch, out_ch):
        super(TopModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=im_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=im_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Conv2d(3, 3, 3, 1, 1)
        self.scale_layer = ScaleLayer()

    def forward(self, x):
        x = self.scale_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.conv(x)
        return x


model = TopModel(im_ch=3, out_ch=3)
model_dict = model.state_dict()
print(list(model_dict.keys()))

print("===")
model_dict = torch.load("D:\python\interactive_keypoint_estimation\code\pretrained_models\hrnetv2_w32_imagenet_pretrained.pth")
for key in list(model_dict.keys()):
    if "conv2" in key:
        print(key)
print(list(model_dict.keys())[-5:-1])
print(f">> num of pretrained HRNet param keys: {len(list(model_dict.keys()))}")

print("===")
model_dict = torch.load("D:\python\interactive_keypoint_estimation\save\ExpNum[00001]_Dataset[dataset16]_Model[RITM_SE_HRNet32]_config[spineweb_ours]_seed[42]\model.pth")
print(list(model_dict["model"].keys()))
print(f">> num of pretrained IKEM param keys: {len(list(model_dict['model'].keys()))}")
