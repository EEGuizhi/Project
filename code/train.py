import os
import math
import random
import datetime
import numpy as np

import cv2
import torch
import torch.nn as nn
import albumentations as A
# keypoints_augmentation：https://albumentations.ai/docs/getting_started/keypoints_augmentation

from model.model import IKEM
from tools.dataset import SpineDataset
from tools.heatmap_maker import HeatmapMaker
from tools.loss import CustomLoss
from tools.dataset import custom_collate_fn
from tools.loss import find_worst_index

"""
當前目標：
1. 設置"NOT_INTERACTIVE"參數，決定模型的訓練方式、模型的輸入、單個epoch的Loss。
2. 以CSV檔儲存Loss相關資料。
"""

# Path Settings
IMAGE_ROOT = "./dataset/dataset16/boostnet_labeldata"
FILE_PATH = "./dataset/all_data.json"
PRETRAINED_MODEL_PATH = "./pretrained_model/hrnetv2_w32_imagenet_pretrained.pth"
CHECKPOINT_PATH = None

# Basic Settings
IMAGE_SIZE = (512, 256)
HEATMAP_STD = 7.5
NUM_OF_KEYPOINTS = 68
USE_CUSTOM_LOSS = True
NOT_INTERACTIVE = False

# Training Settings
EPOCH = 300
BATCH_SIZE = 8
LR = 1e-3


def set_seed(seed):
    '''
    設置相同的隨機種子能確保每次執行結果一致。
    '''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    return

def save_model(path:str, epoch:int, model:nn.Module, optimizer:torch.optim.Optimizer):
    print("Saving model..")
    save_dict = {
        "model": model.state_dict(),
        "epoch": epoch,
        "optimizer": optimizer.state_dict()
    }
    torch.save(save_dict, path)


if __name__ == '__main__':
    # Program Start
    print(f"\n>> Start Program --- {datetime.datetime.now()} \n")
    date = str(datetime.date.today()) + "_0"
    with open("Training_Log_{}.txt".format(date), 'w') as f:
        f.write(f">> Start Program --- {datetime.datetime.now()} \n")

    # Basic settings
    set_seed(42)
    print("Using device: {}".format("cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("Training_Log_{}.txt".format(date), 'a') as f:
        f.write("Using device: {}".format("cuda" if torch.cuda.is_available() else "cpu"))

    # Dataset
    train_transform = A.Compose([
        A.augmentations.geometric.rotate.SafeRotate((-15, 15), p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.augmentations.geometric.resize.RandomScale((-0.1, 0.0), p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.augmentations.geometric.resize.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1], p=1)
    ], keypoint_params=A.KeypointParams(format='yx', remove_invisible=False))
    val_transform = A.Compose([
        A.augmentations.geometric.resize.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1], p=1)
    ], keypoint_params=A.KeypointParams(format='yx', remove_invisible=False))

    train_set = SpineDataset(IMAGE_SIZE, NUM_OF_KEYPOINTS, data_file_path=FILE_PATH, img_root=IMAGE_ROOT, transform=train_transform, set="train")
    val_set = SpineDataset(IMAGE_SIZE, NUM_OF_KEYPOINTS, data_file_path=FILE_PATH, img_root=IMAGE_ROOT, transform=val_transform, set="val")
    train_loader = torch.utils.data.DataLoader(train_set, BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = torch.utils.data.DataLoader(val_set, BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)

    # Initialize
    print("Initialize model...")
    model = IKEM(pretrained_model_path=PRETRAINED_MODEL_PATH).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # https://medium.com/%E9%9B%9E%E9%9B%9E%E8%88%87%E5%85%94%E5%85%94%E7%9A%84%E5%B7%A5%E7%A8%8B%E4%B8%96%E7%95%8C/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92ml-note-sgd-momentum-adagrad-adam-optimizer-f20568c968db
    heatmapMaker = HeatmapMaker(IMAGE_SIZE, HEATMAP_STD)
    loss_func = CustomLoss(use_coord_loss=False, use_angle_loss=True, heatmap_maker=heatmapMaker) if USE_CUSTOM_LOSS else nn.BCELoss()

    if CHECKPOINT_PATH is not None:
        print("Loading model parameters...")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model_param = checkpoint["model"]
        model.load_state_dict(model_param)
        try:
            start_epoch = checkpoint["epoch"] + 1
            optimizer_param = checkpoint["optimizer"]
            optimizer.load_state_dict(optimizer_param)
        except:
            print("Load optimizer state dict failed..")
            start_epoch = 0
        del model_param, optimizer_param, checkpoint
    else:
        start_epoch = 0

    # Calculate the number of model parameters
    n_params = 0
    for k, v in model.named_parameters():  # 遍歷model每一層, k是名稱, v是參數值
        n_params += v.reshape(-1).shape[0]  # v是一個tensor, reshape(-1)表示將v展平; shape[0]表示v展平後的元素個數
    print('Number of model parameters: {}'.format(n_params))

    # Training
    for epoch in range(start_epoch, EPOCH+1):
        print(f"\n>> Epoch：{epoch}")
        model.train()
        train_1stpred_loss = 0
        train_2ndpred_loss = 0
        for i, (images, labels, hint_indexes, y_x_size) in enumerate(train_loader):
            # Init
            images = images.to(device)
            labels = labels.to(device)
            labels_heatmap = heatmapMaker.coord2heatmap(labels).to(dtype=images.dtype)
            hint_heatmap = torch.zeros_like(labels_heatmap)
            prev_pred = torch.zeros_like(hint_heatmap)

            # Determine hint times of this batch during training
            prob = np.array([math.pow(2, -i) for i in range(NUM_OF_KEYPOINTS)])
            prob = prob.tolist() / prob.sum()
            hint_times = np.random.choice(a=NUM_OF_KEYPOINTS, size=None, p=prob)

            # Simulate user interaction
            for click in range(hint_times+1):
                # Model forward
                outputs, aux_out = model(hint_heatmap, prev_pred, images)
                prev_pred = outputs.detach().sigmoid()

                # Update Model
                pred_heatmap = outputs.sigmoid()
                loss = loss_func(pred_heatmap, labels, labels_heatmap) if USE_CUSTOM_LOSS else loss_func(pred_heatmap, labels_heatmap)
                loss += nn.BCELoss()(aux_out.sigmoid(), labels_heatmap)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Inputs update
                keypoints = heatmapMaker.heatmap2sargmax_coord(prev_pred)
                for s in range(hint_heatmap.shape[0]):  # s = idx of samples
                    index = find_worst_index(keypoints[s], labels[s])
                    hint_heatmap[s, index] = labels_heatmap[s, index]

                # Loss log
                if click == 0: train_1stpred_loss += loss.item() / len(train_loader)
                elif click == 1: train_2ndpred_loss += loss.item() / len(train_loader)
        print(f"Training (first pred.) Loss：{round(train_1stpred_loss, 3)}")
        print(f"Training (second pred.) Loss：{round(train_2ndpred_loss, 3)}")
        with open("Training_Log_{}.txt".format(date), 'a') as f:
            f.write(f"\n>> Epoch：{epoch}     ")
            f.write(f"Training (first pred.) Loss：{round(train_1stpred_loss, 3)}  ")
            f.write(f"Training (second pred.) Loss：{round(train_2ndpred_loss, 3)}     ")

        model.eval()
        val_1stloss = 0
        val_2ndloss = 0
        with torch.no_grad():
            for i, (inputs, labels, hint_indexes, y_x_size) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                labels_heatmap = heatmapMaker.coord2heatmap(labels)
                hint_heatmap = torch.zeros_like(labels_heatmap)
                prev_pred = torch.zeros_like(hint_heatmap)

                for click in range(2):
                    outputs, aux_out = model(hint_heatmap, prev_pred, images)
                    prev_pred = outputs.detach().sigmoid()

                    pred_heatmap = outputs.sigmoid()
                    loss = loss_func(pred_heatmap, labels, labels_heatmap) if USE_CUSTOM_LOSS else loss_func(pred_heatmap, labels_heatmap)
                    loss += nn.BCELoss()(aux_out.sigmoid(), labels_heatmap)

                    # Inputs update
                    keypoints = heatmapMaker.heatmap2sargmax_coord(prev_pred)
                    for s in range(hint_heatmap.shape[0]):  # s = idx of samples
                        index = find_worst_index(keypoints[s], labels[s])
                        hint_heatmap[s, index] = labels_heatmap[s, index]

                    if click == 0: val_1stloss += loss.item() / len(val_loader)
                    elif click == 1: val_2ndloss += loss.item() / len(val_loader)
        print(f"Validation (first pred.) Loss：{round(val_1stloss, 3)}")
        print(f"Validation (second pred.) Loss：{round(val_2ndloss, 3)}")
        with open("Training_Log_{}.txt".format(date), 'a') as f:
            f.write(f"Validation (first pred.) Loss：{round(val_1stloss, 3)}  ")
            f.write(f"Validation (second pred.) Loss：{round(val_2ndloss, 3)}")

        save_model("checkpoint_{}.pth".format(epoch//50), epoch, model, optimizer)

    # Program Ended
    print(f"\n>> End Program --- {datetime.datetime.now()} \n")
    with open("Training_Log_{}.txt".format(date), 'a') as f:
        f.write(f"\n>> End Program --- {datetime.datetime.now()} \n")
