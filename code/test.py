import os
import datetime

import torch
import albumentations as A  # keypoints_augmentation：https://albumentations.ai/docs/getting_started/keypoints_augmentation

from model.model import IKEM
from model.unet_IKEM import UNet_IKEM
from tools.dataset import SpineDataset
from tools.heatmap_maker import HeatmapMaker
from tools.dataset import custom_collate_fn
from tools.misc import *


# Model
MODELS = ["HRNetOCR_IKEM", "UNet_IKEM"]
USE_MODEL = 0
WITH_IGGN = True

# Path settings
IMAGE_ROOT = "./dataset/dataset16/boostnet_labeldata"
CHECKPOINT_PATH = ""
FILE_PATH = "./dataset/all_data.json"

# Basic settings
HINT_TIMES = 5
IMAGE_SIZE = (512, 256)
HEATMAP_STD = 7.5
NUM_OF_KEYPOINTS = 68
BATCH_SIZE = 8


if __name__ == '__main__':
    # Program Start
    print(f"\n>> Start Program --- {datetime.datetime.now()} \n")

    # Basic settings
    set_seed(42)
    print("Using device: {}".format("cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    test_transform = A.Compose([
        A.augmentations.geometric.resize.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1], p=1)
    ], keypoint_params=A.KeypointParams(format='yx', remove_invisible=False))
    test_set = SpineDataset(IMAGE_SIZE, NUM_OF_KEYPOINTS, data_file_path=FILE_PATH, img_root=IMAGE_ROOT, transform=test_transform, set="test")
    test_loader = torch.utils.data.DataLoader(test_set, BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)

    # Initialize
    print(f"\nUsing Model: {MODELS[USE_MODEL]}")
    print("Initialize model...")
    if MODELS[USE_MODEL] == "HRNetOCR_IKEM":
        model = IKEM(use_iggnet=WITH_IGGN, pretrained_model_path=None).to(device)
    elif MODELS[USE_MODEL] == "UNet_IKEM":
        model = UNet_IKEM(use_iggnet=WITH_IGGN).to(device)
    heatmapMaker = HeatmapMaker(IMAGE_SIZE, HEATMAP_STD)

    if CHECKPOINT_PATH is not None:
        print("Loading model parameters...")
        checkpoint = torch.load(CHECKPOINT_PATH)
        epoch = checkpoint["epoch"]
        model_param = checkpoint["model"]
        model.load_state_dict(model_param)
        del model_param, checkpoint
    else:
        print("Need checkpoint file to start testing..")
        exit()

    # Calculate the number of model parameters
    n_params = 0
    for k, v in model.named_parameters():  # 遍歷model每一層, k是名稱, v是參數值
        n_params += v.reshape(-1).shape[0]  # v是一個tensor, reshape(-1)表示將v展平; shape[0]表示v展平後的元素個數
    print("Number of model parameters: {}".format(n_params))

    print(f"\nCheckpoint Model is trained after {epoch} epoch\n")

    # Testing
    sample_count = 0
    Model_MRE = [0 for i in range(HINT_TIMES+1)]
    Manual_MRE = [0 for i in range(HINT_TIMES+1)]
    with torch.no_grad():
        model.eval()
        for i, (images, labels, hint_indexes, y_x_size) in enumerate(test_loader):
            # Init
            images = images.to(device)
            labels = labels.to(device)
            y_x_size = y_x_size.to(device)
            image_size = torch.tensor(IMAGE_SIZE, device=device)

            # Get coords in original size
            keypoints = torch.zeros_like(labels)
            scale_manual_keypoints = torch.zeros_like(labels)
            origSize_labels = torch.zeros_like(labels)
            for s in range(labels.shape[0]):
                origSize_labels[s] = labels[s] * y_x_size[s] / image_size

            labels_heatmap = heatmapMaker.coord2heatmap(labels).to(dtype=images.dtype)
            hint_heatmap = torch.zeros_like(labels_heatmap)
            prev_pred = torch.zeros_like(hint_heatmap)

            # Simulate user interaction
            manual_revision = [[] for s in range(labels.shape[0])]
            for click in range(HINT_TIMES+1):
                # Model forward
                if MODELS[USE_MODEL] == "HRNetOCR_IKEM":
                    outputs, aux_out = model(hint_heatmap, prev_pred, images)
                elif MODELS[USE_MODEL] == "UNet_IKEM":
                    outputs = model(hint_heatmap, prev_pred, images)
                prev_pred = outputs.sigmoid()

                pred_coords = heatmapMaker.heatmap2sargmax_coord(prev_pred)
                if click == 0: manual_keypoints = pred_coords.clone()

                # Get MRE
                for s in range(hint_heatmap.shape[0]):
                    # Number of samples
                    if click == 0: sample_count += 1

                    # Scale to original size
                    keypoints[s] = pred_coords[s] * y_x_size[s] / image_size
                    scale_manual_keypoints[s] = manual_keypoints[s] * y_x_size[s] / image_size

                    # Calc MRE
                    for item in manual_revision[s]:
                        keypoints[s, item["index"]] = item["coord"]
                    Model_MRE[click] += get_MRE(keypoints[s], origSize_labels[s])
                    Manual_MRE[click] += get_MRE(scale_manual_keypoints[s], origSize_labels[s])

                    # Model revision
                    index = find_worst_index(pred_coords[s], labels[s])
                    hint_heatmap[s, index] = labels_heatmap[s, index]
                    manual_revision[s].append({
                        "index": index,
                        "coord": origSize_labels[s, index]
                    })

                    # Manual revision
                    index = find_worst_index(manual_keypoints[s], labels[s])
                    manual_keypoints[s, index] = labels[s, index]

    # Outputs
    print("[Model Revision]")
    for i in range(HINT_TIMES+1):
        print(f"Mean Radial Error (click {i}): {Model_MRE[i] / sample_count}")

    print("\n[Fully Manual Revision]")
    for i in range(HINT_TIMES+1):
        print(f"Mean Radial Error (click {i}): {Manual_MRE[i] / sample_count}")


    # Program Ended
    print(f"\n>> End Program --- {datetime.datetime.now()} \n")
