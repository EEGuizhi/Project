import os
import math
import datetime
import numpy as np

import cv2
import torch
import torch.nn as nn
import albumentations as A  # keypoints_augmentation：https://albumentations.ai/docs/getting_started/keypoints_augmentation

from model.hrnet_ocr import hrnet_ocr
from tools.dataset import SpineDataset
from tools.heatmap_maker import HeatmapMaker
from tools.loss import CustomLoss
from tools.dataset import custom_collate_fn
from tools.misc import *


# Path Settings
IMAGE_ROOT = ""
FILE_PATH = "./dataset/all_data.json"
PRETRAINED_MODEL_PATH = "./pretrained_model/hrnetv2_w32_imagenet_pretrained.pth"
CHECKPOINT_PATH = ""
TARGET_FOLDER = ""

# Basic Settings
IMAGE_SIZE = (512, 256)
HEATMAP_STD = 7.5
NUM_OF_KEYPOINTS = 68

USE_CUSTOM_LOSS = False

# Training Settings
EPOCH = 249
BATCH_SIZE = 8
LR = 1e-3


if __name__ == '__main__':
    # Program Start
    print(f"\n>> Start Program --- {datetime.datetime.now()} \n")
    date = str(datetime.date.today()) + "_0"

    # Basic settings
    set_seed(42)
    print("Using device: {}".format("cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    train_transform = A.Compose([
        A.augmentations.geometric.rotate.SafeRotate((-10, 10), p=0.5, border_mode=cv2.BORDER_CONSTANT),
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
    model = hrnet_ocr(pretrained_model_path=PRETRAINED_MODEL_PATH).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # https://medium.com/%E9%9B%9E%E9%9B%9E%E8%88%87%E5%85%94%E5%85%94%E7%9A%84%E5%B7%A5%E7%A8%8B%E4%B8%96%E7%95%8C/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92ml-note-sgd-momentum-adagrad-adam-optimizer-f20568c968db
    heatmapMaker = HeatmapMaker(IMAGE_SIZE, HEATMAP_STD)
    loss_func = CustomLoss(use_coord_loss=False, use_angle_loss=True) if USE_CUSTOM_LOSS else nn.BCELoss()

    if CHECKPOINT_PATH is not None:
        print("Loading model parameters...")
        checkpoint = torch.load(CHECKPOINT_PATH)
        start_epoch = checkpoint["epoch"] + 1
        model_param = checkpoint["model"]
        model.load_state_dict(model_param)
        optimizer_param = checkpoint["optimizer"]
        optimizer.load_state_dict(optimizer_param)
        try:
            saved_train_loss, saved_val_MRE = checkpoint["train_loss"], checkpoint["val_MRE"]
        except:
            saved_train_loss, saved_val_MRE = None, None
        del model_param, optimizer_param, checkpoint
    else:
        saved_train_loss, saved_val_MRE = None, None
        start_epoch = 1

    # Calculate the number of model parameters
    n_params = 0
    for k, v in model.named_parameters():
        n_params += v.reshape(-1).shape[0]
    print(f"Number of model parameters: {n_params}")

    # Epoch
    dataframe, lowest_MRE, epoch_count = None, None, None
    for epoch in range(start_epoch, EPOCH+1):
        print(f"\n>> Epoch：{epoch}")

        # Training
        train_loss = 0
        for i, (images, labels, hint_indexes, y_x_size) in enumerate(train_loader):
            # Init
            images = images.to(device)
            labels = labels.to(device)
            labels_heatmap = heatmapMaker.coord2heatmap(labels).to(dtype=images.dtype)

            # Train model
            model.train()
            outputs, aux_out = model(images)  # Model forward

            # Update Model
            pred_heatmap = outputs.sigmoid()
            pred_coord = heatmapMaker.heatmap2sargmax_coord(pred_heatmap)
            loss = loss_func(pred_coord, pred_heatmap, labels, labels_heatmap) if USE_CUSTOM_LOSS else loss_func(pred_heatmap, labels_heatmap)
            loss += nn.BCELoss()(aux_out.sigmoid(), labels_heatmap)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss += loss.item() / len(train_loader)
        print(f"Training Loss：{round(train_loss, 3)}")


        # Validation
        val_loss = 0
        val_MRE = 0
        model.eval()
        with torch.no_grad():
            for i, (images, labels, hint_indexes, y_x_size) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                labels_heatmap = heatmapMaker.coord2heatmap(labels)

                outputs, aux_out = model(images)
                pred_heatmap = outputs.sigmoid()
                pred_coord = heatmapMaker.heatmap2sargmax_coord(pred_heatmap)
                loss = loss_func(pred_coord, pred_heatmap, labels, labels_heatmap) if USE_CUSTOM_LOSS else loss_func(pred_heatmap, labels_heatmap)
                loss += nn.BCELoss()(aux_out.sigmoid(), labels_heatmap)

                # Inputs update
                val_loss += loss.item()
                val_MRE += get_batch_MRE(pred_coord, labels).item()

        val_loss = val_loss / len(val_loader)
        val_MRE = val_MRE / len(val_loader)
        print(f"Validation Loss：{round(val_loss, 3)}")
        print(f"Validation MRE = {round(val_MRE, 3)}")

        # Saving data
        stop_training, lowest_MRE, epoch_count = early_stop(val_MRE, lowest_MRE, epoch_count)
        dataframe = write_log(
            os.path.join(TARGET_FOLDER, f"Training_Log_{date}.csv"),
            dataframe, epoch, train_loss, None,
            val_loss, None, val_MRE, None
        )
        better_pred, larger_gap, saved_train_loss, saved_val_MRE = is_worth_to_save(
            train_loss=(train_loss, None), val_MRE=(val_MRE, None),
            saved_train_loss=saved_train_loss, saved_val_MRE=saved_val_MRE
        )
        if better_pred:
            save_model(
                os.path.join(TARGET_FOLDER, f"Checkpoint_HRNetOCR_BestPred.pth"),
                epoch, model, optimizer, saved_train_loss, saved_val_MRE, msg="BestPred"
            )
        save_model(
            os.path.join(TARGET_FOLDER, f"Checkpoint_HRNetOCR_Newest.pth"),
            epoch, model, optimizer, saved_train_loss, saved_val_MRE
        )

        if stop_training is True:
            print(f"Early stop at epoch {epoch}")
            break

    # Program Ended
    print(f"\n>> End Program --- {datetime.datetime.now()} \n")
