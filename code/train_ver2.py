import os
import math
import datetime
import numpy as np

import cv2
import torch
import torch.nn as nn
import albumentations as A  # keypoints_augmentation：https://albumentations.ai/docs/getting_started/keypoints_augmentation

from model.model import IKEM
from model.unet_IKEM import UNet_IKEM
from tools.dataset import SpineDataset
from tools.heatmap_maker import HeatmapMaker
from tools.loss import CustomLoss
from tools.dataset import custom_collate_fn
from tools.misc import *

"""
[train_ver2.py] :
Using hint heatmap before first prediction,
simulate user interaction (update hint heatmap) in each iteration.
"""


# Model
MODELS = ["HRNetOCR_IKEM", "UNet_IKEM"]
USE_MODEL = 0
WITH_IGGN = True  # interaction-guided dating network

# Path Settings
IMAGE_ROOT = "./dataset/dataset16/boostnet_labeldata"
FILE_PATH = "./dataset/all_data.json"
PRETRAINED_MODEL_PATH = "./pretrained_model/hrnetv2_w32_imagenet_pretrained.pth"
TARGET_FOLDER = ""
CHECKPOINT_PATH = None

# Basic Settings
IMAGE_SIZE = (512, 256)
HEATMAP_STD = 7.5
NUM_OF_KEYPOINTS = 68

USE_CUSTOM_LOSS = False
MAX_HINT_TIMES = 10
MAX_ITER_TIMES = 3
HINT_TRAINING_AFTER_EPOCH = 20

# Training Settings
EPOCH = 199
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
        A.augmentations.geometric.rotate.SafeRotate((-15, 15), p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.augmentations.geometric.resize.RandomScale((-0.1, 0.2), p=0.5),
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
    print(f"\nUsing Model: {MODELS[USE_MODEL]}")
    print("Initialize model...")
    if MODELS[USE_MODEL] == "HRNetOCR_IKEM":
        model = IKEM(use_iggnet=WITH_IGGN, pretrained_model_path=PRETRAINED_MODEL_PATH).to(device)
    elif MODELS[USE_MODEL] == "UNet_IKEM":
        model = UNet_IKEM(use_iggnet=WITH_IGGN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # https://medium.com/%E9%9B%9E%E9%9B%9E%E8%88%87%E5%85%94%E5%85%94%E7%9A%84%E5%B7%A5%E7%A8%8B%E4%B8%96%E7%95%8C/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92ml-note-sgd-momentum-adagrad-adam-optimizer-f20568c968db
    heatmapMaker = HeatmapMaker(IMAGE_SIZE, HEATMAP_STD)
    loss_func = CustomLoss(use_coord_loss=False, use_morph_loss=True) if USE_CUSTOM_LOSS else nn.BCELoss()

    if CHECKPOINT_PATH is not None:
        print("Loading model parameters...")
        checkpoint = torch.load(CHECKPOINT_PATH)
        start_epoch = checkpoint["epoch"] + 1
        model_param = checkpoint["model"]
        model.load_state_dict(model_param)
        optimizer_param = checkpoint["optimizer"]
        optimizer.load_state_dict(optimizer_param)
        try:
            lowest_MRE, epoch_count = checkpoint["lowest_MRE"], checkpoint["epoch_count"]
        except:
            lowest_MRE, epoch_count = None, None
        del model_param, optimizer_param, checkpoint
    else:
        lowest_MRE, epoch_count = None, None
        start_epoch = 1

    # Calculate the number of model parameters
    n_params = 0
    for k, v in model.named_parameters():
        n_params += v.reshape(-1).shape[0]
    print(f"Number of model parameters: {n_params}")


    # Epoch
    dataframe = None
    for epoch in range(start_epoch, EPOCH+1):
        print(f"\n>> Epoch：{epoch}")

        # Training
        pred1_loss = []
        pred2_loss = []
        for i, (images, labels, hint_indexes, y_x_size) in enumerate(train_loader):
            # Init
            images = images.to(device)
            labels = labels.to(device)
            labels_heatmap = heatmapMaker.coord2heatmap(labels).to(dtype=images.dtype)
            hint_heatmap = torch.zeros_like(labels_heatmap)
            prev_pred = torch.zeros_like(hint_heatmap)

            # Determine hint times of this batch during training
            if epoch <= HINT_TRAINING_AFTER_EPOCH:
                hint_times = 0
            else:
                prob = np.array([math.pow(2, -i) for i in range(MAX_HINT_TIMES+1)])
                prob[0] = prob[1] / 2
                prob = prob.tolist() / prob.sum()
                hint_times = np.random.choice(a=MAX_HINT_TIMES+1, size=None, p=prob)

            # Update hint heatmap before first pred.
            if hint_times > 0:
                for s in range(hint_heatmap.shape[0]):
                    hint_heatmap[s, hint_indexes[s, 0:hint_times]] = labels_heatmap[s, hint_indexes[s, 0:hint_times]]

            # Iterztive training
            iter_times = np.random.choice(MAX_ITER_TIMES, size=None)
            model.eval()
            with torch.no_grad():
                for iter in range(iter_times):
                    # Model forward
                    if MODELS[USE_MODEL] == "HRNetOCR_IKEM":
                        outputs, aux_out = model(hint_heatmap, prev_pred, images)
                    elif MODELS[USE_MODEL] == "UNet_IKEM":
                        outputs = model(hint_heatmap, prev_pred, images)
                    prev_pred = outputs.detach().sigmoid()

                    # Inputs update
                    pred_coord = heatmapMaker.heatmap2expected_coord(prev_pred)
                    for s in range(hint_heatmap.shape[0]):  # s = idx of samples
                        index = find_worst_index(pred_coord[s], labels[s])
                        hint_heatmap[s, index] = labels_heatmap[s, index]

            # Train model
            model.train()
            if MODELS[USE_MODEL] == "HRNetOCR_IKEM":
                outputs, aux_out = model(hint_heatmap, prev_pred, images)
            elif MODELS[USE_MODEL] == "UNet_IKEM":
                outputs = model(hint_heatmap, prev_pred, images)

            # Update Model
            pred_heatmap = outputs.sigmoid()
            pred_coord = heatmapMaker.heatmap2expected_coord(pred_heatmap)
            loss = loss_func(pred_coord, pred_heatmap, labels, labels_heatmap) if USE_CUSTOM_LOSS else loss_func(pred_heatmap, labels_heatmap)
            if MODELS[USE_MODEL] == "HRNetOCR_IKEM": loss += nn.BCELoss()(aux_out.sigmoid(), labels_heatmap)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Loss log
            if hint_times == 0: pred1_loss.append(loss.item())
            elif hint_times == 1: pred2_loss.append(loss.item())

        train_p1Loss = np.array(pred1_loss).sum() / len(pred1_loss)
        print(f"Training (first pred.) Loss：{round(train_p1Loss, 3)}")
        if len(pred2_loss) > 0:
            train_p2Loss = np.array(pred2_loss).sum() / len(pred2_loss)
            print(f"Training (second pred.) Loss：{round(train_p2Loss, 3)}")
        else:
            train_p2Loss = None


        # Validation
        pred1_loss = []
        pred2_loss = []
        pred1_MRE = []
        pred2_MRE = []
        model.eval()
        with torch.no_grad():
            for i, (images, labels, hint_indexes, y_x_size) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                labels_heatmap = heatmapMaker.coord2heatmap(labels)
                hint_heatmap = torch.zeros_like(labels_heatmap)
                prev_pred = torch.zeros_like(hint_heatmap)

                for click in range(2):
                    if MODELS[USE_MODEL] == "HRNetOCR_IKEM":
                        outputs, aux_out = model(hint_heatmap, prev_pred, images)
                    elif MODELS[USE_MODEL] == "UNet_IKEM":
                        outputs = model(hint_heatmap, prev_pred, images)
                    prev_pred = outputs.detach().sigmoid()

                    pred_heatmap = outputs.sigmoid()
                    pred_coord = heatmapMaker.heatmap2expected_coord(prev_pred)
                    loss = loss_func(pred_coord, pred_heatmap, labels, labels_heatmap) if USE_CUSTOM_LOSS else loss_func(pred_heatmap, labels_heatmap)
                    if MODELS[USE_MODEL] == "HRNetOCR_IKEM": loss += nn.BCELoss()(aux_out.sigmoid(), labels_heatmap)

                    # Inputs update
                    for s in range(hint_heatmap.shape[0]):  # s = idx of samples
                        index = choose_hint_index(pred_coord[s], labels[s])
                        hint_heatmap[s, index] = labels_heatmap[s, index]

                    if click == 0:
                        pred1_loss.append(loss.item())
                        pred1_MRE.append(get_batch_MRE(pred_coord, labels))
                    elif click == 1:
                        pred2_loss.append(loss.item())
                        pred2_MRE.append(get_batch_MRE(pred_coord, labels))

        val_p1Loss = np.array(pred1_loss).sum() / len(pred1_loss)
        val_p2Loss = np.array(pred2_loss).sum() / len(pred2_loss)
        val_p1MRE = np.array(pred1_MRE).sum() / len(pred1_MRE)
        val_p2MRE = np.array(pred2_MRE).sum() / len(pred2_MRE)
        print(f"Validation (first pred.) Loss：{round(val_p1Loss, 3)}")
        print(f"Validation (second pred.) Loss：{round(val_p2Loss, 3)}")
        print(f"Validation pred1 MRE = {round(val_p1MRE, 3)},  pred2 MRE = {round(val_p2MRE, 3)}")

        # Saving data
        dataframe = write_log(
            os.path.join(TARGET_FOLDER, f"Training_iter_Log_{date}.csv"),
            dataframe, epoch, train_p1Loss, train_p2Loss,
            val_p1Loss, val_p2Loss, val_p1MRE, val_p2MRE
        )
        stop_training, lowest_MRE, epoch_count = early_stop(val_p2MRE, lowest_MRE, epoch_count)
        is_best_pred = True if epoch_count == 0 else False
        if is_best_pred:
            save_model(
                os.path.join(TARGET_FOLDER, f"Checkpoint_BestPred.pth"),
                epoch, model, optimizer, lowest_MRE=lowest_MRE, epoch_count=epoch_count, msg="BestPred"
            )
        save_model(
            os.path.join(TARGET_FOLDER, f"Checkpoint_Newest.pth"),
            epoch, model, optimizer, lowest_MRE=lowest_MRE, epoch_count=epoch_count
        )

        if stop_training is True:
            print(f"Early stop at epoch {epoch}")
            break

    # Program Ended
    print(f"\n>> End Program --- {datetime.datetime.now()} \n")
