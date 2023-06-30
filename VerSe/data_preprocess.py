import os
import cv2
import json
import math
import numpy as np
import nibabel as nib
import nibabel.orientations as nio
import matplotlib.pyplot as plt

from data_utilities import *


# # 讀取圖像
# nii_file = nib.load("D:/GitHub/EEGuizhi/Project/VerSe/dataset-01training/derivatives/sub-verse500/sub-verse500_dir-ax_seg-vert_msk.nii.gz")
# nii_file = resample_nib(nii_file, voxel_spacing=(1, 1, 1), order=0)
# img_data = nii_file.get_fdata()

# print(nii_file)
# print(nii_file.header.get_zooms())

# # 讀取各節脊柱中心座標
# JSON_FILE = "D:/GitHub/EEGuizhi/Project/VerSe/dataset-01training/derivatives/sub-verse500/sub-verse500_dir-ax_seg-subreg_ctd.json"
# center_coords = []
# with open(JSON_FILE) as f:
#     json_data = json.load(f)
#     for data in json_data:
#         if "label" in data:
#             center_coords.append([
#                 int(data["X"]),  # 'X':右身到左身
#                 int(data["Y"]),  # 'Y':背到胸
#                 int(data["Z"])   # 'Z':頸到盆
#             ])

# ### 輸出圖像
# projection = np.sum(img_data[:, 180:, :], axis=1)  # 沿axis=0:側身軸, axis=1:胸到背, axis=2:俯視
# plt.imsave("TestOutput_withoutkeypoints.png", projection, cmap="gray")

# color_img = cv2.imread("TestOutput_withoutkeypoints.png")
# for coord in center_coords:
#     cv2.circle(color_img, (coord[2], coord[0]), 3, (0, 0, 255), -1)
# plt.imsave("TestOutput_withkeypoints.png", color_img, cmap="gray")


# # 將圖片轉成灰階
# color_img = cv2.imread("TestOutput_withoutkeypoints.png")
# gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
# gray = cv2.blur(gray, (3, 3))

# # 將圖像標準化到 (-5, 5) 範圍内
# gray_norm = (gray - np.mean(gray)) / np.std(gray)

# # 使用 sigmoid 函數對亮度進行調整
# sigmoid = lambda x: 1 / (1 + np.exp(-x))
# gray = 255 * sigmoid(10 * (gray_norm - 0.5))
# gray = np.array(gray, dtype=np.uint8)

# # 設定角點偵測的參數
# block_size = 7
# ksize = 5
# k = 0.04
# threshold = 0.3

# # 使用cv2.cornerHarris()方法進行角點偵測
# dst = cv2.cornerHarris(gray, block_size, ksize, k)
# dst = cv2.dilate(dst, None)
# feature_points = []
# threshold = threshold * dst.max()
# for x in range(dst.shape[0]):
#     for y in range(dst.shape[1]):
#         if dst[x, y] > threshold:
#             feature_points.append([x, y])

# def merge_close_points(coords, r):
#     """將距離小於 r 的座標平均起來"""
#     points = np.array(coords)
#     merged_points = []  # 儲存合併後的座標
#     num_points = len(points)
#     visited = np.zeros(num_points)  # 標記已經合併的座標

#     for i in range(num_points):
#         if visited[i]:  # 如果已經合併過，跳過
#             continue
        
#         # 找到所有距離小於 r 的座標
#         merge_indices = [i]
#         for j in range(i+1, num_points):
#             if visited[j]:  # 如果已經合併過，跳過
#                 continue
#             dist = np.linalg.norm(np.array(points[i]) - np.array(points[j]))  # 計算距離
#             if dist < r:
#                 merge_indices.append(j)
        
#         # 計算這些座標的平均值
#         merged_point = np.mean(np.array(points[merge_indices]), axis=0)
#         merged_points.append(merged_point.astype(int))  # 將平均座標轉換為整數座標
#         visited[merge_indices] = 1  # 標記已經合併的座標

#     return merged_points

# true_points = merge_close_points(feature_points, 8)
# true_points = merge_close_points(true_points, 8)

# # 輸出圖像
# for coord in true_points:
#     cv2.circle(color_img, (coord[1], coord[0]), 3, (0, 0, 255), -1)
# plt.imsave("TestOutput2_withkeypoints.png", color_img, cmap="gray")



### New Codes ###

# data directory
MSK_PATH = "D:/GitHub/EEGuizhi/Project/VerSe/dataset-01training/derivatives/sub-verse500/sub-verse500_dir-ax_seg-vert_msk.nii.gz"
IMG_PATH = "D:/GitHub/EEGuizhi/Project/VerSe/dataset-01training/rawdata/sub-verse500/sub-verse500_dir-ax_ct.nii.gz"
CTD_PATH = "D:/GitHub/EEGuizhi/Project/VerSe/dataset-01training/derivatives/sub-verse500/sub-verse500_dir-ax_seg-subreg_ctd.json"

# load files
img_nib = nib.load(IMG_PATH)
msk_nib = nib.load(MSK_PATH)
ctd_list = load_centroids(CTD_PATH)  # 'X':右身到左身, 'Y':背到胸, 'Z':頸到盆



# check img zooms
zooms = img_nib.header.get_zooms()
print('img zooms = {}'.format(zooms))

# check img orientation
axs_code = nio.ornt2axcodes(nio.io_orientation(img_nib.affine))
print('img orientation code: {}'.format(axs_code))

# check centroids
print('Centroid List: {}'.format(ctd_list))



# Resample and Reorient data
img_iso = resample_nib(img_nib, voxel_spacing=(1, 1, 1), order=3)
msk_iso = resample_nib(msk_nib, voxel_spacing=(1, 1, 1), order=0)  # or resample based on img: resample_mask_to(msk_nib, img_iso)
ctd_iso = rescale_centroids(ctd_list, img_nib, (1,1,1))

img_iso = reorient_to(img_iso, axcodes_to=('I', 'P', 'L'))
msk_iso = reorient_to(msk_iso, axcodes_to=('I', 'P', 'L'))
ctd_iso = reorient_centroids_to(ctd_iso, img_iso)

# check img zooms
zooms = img_iso.header.get_zooms()
print('img zooms = {}'.format(zooms))

# check img orientation
axs_code = nio.ornt2axcodes(nio.io_orientation(img_iso.affine))
print('img orientation code: {}'.format(axs_code))

# check centroids
print('new centroids: {}'.format(ctd_iso))



# get vocel data
img_np  = img_iso.get_fdata()
msk_np = msk_iso.get_fdata()

# 找到適當的分界點
lookFromSide = msk_np[:, :, msk_np.shape[2]//2]

def find_split_point(img):
    Down_flag = False
    Up_flag = False
    sums = [0, 0, 0, 0]
    for x in range(img.shape[1]):  # 從左到右 一縱行一縱行的計算
        sums[0] = sums[1]
        sums[1] = sums[2]
        sums[2] = sums[3]
        sums[3] = 0
        for y in range(img.shape[0]):
            sums[3] += img[y, x]
        
        if int(np.mean(sums[2:3])) > int(np.mean(sums[0:2])):
            if not Up_flag:
                Up_flag = True
            elif Down_flag:
                return x
        elif Up_flag and int(np.mean(sums[2:3])) < int(np.mean(sums[0:2]) * 0.9):
            Down_flag = True
    return 0

# split_index = find_split_point(lookFromSide)
split_index = msk_np.shape[1]
print(split_index)

raw_proj = np.sum(img_np[:, :, :], axis=1)
msk_proj = np.sum(msk_np[:, :split_index, :], axis=1)

# 儲存結果為.png檔案
plt.imsave('msk_image.png', msk_proj, cmap='gray')
plt.imsave('raw_image.png', raw_proj, cmap='gray')
