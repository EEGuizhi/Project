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


# ### nii file
# import nibabel as nib
# import matplotlib.pyplot as plt

# # 載入NIfTI檔案
# # nii_img = nib.load('D:/python/volume-covid19-A-0025_ct_seg.nii')
# nii_img = nib.load('D:/GitHub/EEGuizhi/Project/new_example.nii')

# # 取得影像數據
# img_data = nii_img.get_fdata()

# # 顯示影像
# plt.imshow(img_data[:, :, 5], cmap='gray')
# plt.show()


# ### nii file 2
# import nibabel as nib
# import matplotlib.pyplot as plt

# # 讀取NIfTI檔案
# # nii_file = nib.load('D:/python/liver_96_seg.nii')
# nii_file = nib.load("D:/python/CTSpine1K-20230410T033504Z-001/CTSpine1K/completed_annotation_verse/verse074_seg.nii.gz")
# img_data = nii_file.get_fdata()

# # 創建一個3x1的子圖畫板
# fig, axes = plt.subplots(nrows=3, ncols=1)

# # 在不同的子圖中顯示從不同軸向的切片
# axes[0].imshow(img_data[:, :, img_data.shape[2]//2], cmap='gray')
# axes[0].set_title('Axial plane')
# axes[1].imshow(img_data[:, img_data.shape[1]//2, :], cmap='gray')
# axes[1].set_title('Coronal plane')
# axes[2].imshow(img_data[img_data.shape[0]//2, :, :], cmap='gray')
# axes[2].set_title('Sagittal plane')

# # 顯示子圖畫板
# plt.show()

# ### nii file 3
# import matplotlib
# matplotlib.use('TkAgg')
 
# from matplotlib import pylab as plt
# import nibabel as nib
# from nibabel import nifti1
# from nibabel.viewers import OrthoSlicer3D
 
# example_filename = "D:/python/CTSpine1K-20230410T033504Z-001/CTSpine1K/completed_annotation_verse/verse074_seg.nii.gz"
 
# img = nib.load(example_filename)
# print (img)
# print (img.header['db_name'])   # 輸出頭信息
 
# width,height,queue=img.dataobj.shape
 
# OrthoSlicer3D(img.dataobj).show()
 
# num = 1
# for i in range(0,queue,10):
 
#     img_arr = img.dataobj[:,:,i]
#     plt.subplot(5,4,num)
#     plt.imshow(img_arr,cmap='gray')
#     num +=1
 
# plt.show()



# ### 開啟.raw檔
# import rawpy
# import imageio

# path = "C:/Users/danie/Downloads/Dataset15/trainingData/case1.raw"
# with rawpy.imread(path) as raw:
#     rgb = raw.postprocess()
# imageio.imsave('default.tiff', rgb)


# ### tensor to list
# a = torch.randn(3, 3, 3)
# print(a)
# tmp = a[0, :, :].tolist()
# print(tmp)


### 
# from munch import Munch
# test_dict = Munch.fromDict({})
# test_dict.is_training = True
# test_dict = Munch.toDict(test_dict)
# print(test_dict)


###
# import argparse
# parser = argparse.ArgumentParser(description='TMI experiments')  # 創建
# args = parser.parse_args()
# args.seed = 42
# print(args.seed, type(args))


###
a = Tensor([1, 2, 3])
a = torch.stack([a, a, a])
print(a)

a = torch.tensor([[1]])
a = torch.cat((a, a), 1)
print(a)

a = [1, 2]
a.append(3)
print(a)
