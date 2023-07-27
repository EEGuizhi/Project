import os
import datetime

import torch
import albumentations as A  # keypoints_augmentation：https://albumentations.ai/docs/getting_started/keypoints_augmentation

from model.model import IKEM
from tools.dataset import SpineDataset
from tools.heatmap_maker import HeatmapMaker
from tools.dataset import custom_collate_fn
from tools.misc import *


IMAGE_ROOT = "./dataset/dataset16/boostnet_labeldata"
CHECKPOINT_PATH = ""
FILE_PATH = "./dataset/all_data.json"

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
    print("Initialize model...")
    model = IKEM(pretrained_model_path=None).to(device)
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
    hint_times = 10
    Mean_Radial_Error = [0 for i in range(hint_times+1)]
    with torch.no_grad():
        model.eval()
        for i, (images, labels, hint_indexes, y_x_size) in enumerate(test_loader):
            # Init
            images = images.to(device)
            labels = labels.to(device)
            labels_heatmap = heatmapMaker.coord2heatmap(labels).to(dtype=images.dtype)
            hint_heatmap = torch.zeros_like(labels_heatmap)
            prev_pred = torch.zeros_like(hint_heatmap)

            # Simulate user interaction
            for click in range(hint_times+1):
                # Model forward
                outputs, aux_out = model(hint_heatmap, prev_pred, images)
                prev_pred = outputs.sigmoid()

                # Inputs update
                keypoints = heatmapMaker.heatmap2sargmax_coord(prev_pred)
                for s in range(hint_heatmap.shape[0]):  # s = idx of samples
                    index = find_worst_index(keypoints[s], labels[s])
                    hint_heatmap[s, index] = labels_heatmap[s, index]

                # Get MRE
                for s in range(hint_heatmap.shape[0]):
                    if click == 0: sample_count += 1
                    labels_coord = torch.zeros_like(labels)
                    labels_coord[s, :, 0] = labels[s, :, 0] * y_x_size[s, 0].item() / IMAGE_SIZE[0]
                    labels_coord[s, :, 1] = labels[s, :, 1] * y_x_size[s, 1].item() / IMAGE_SIZE[1]
                    keypoints[s, :, 0] = keypoints[s, :, 0] * y_x_size[s, 0].item() / IMAGE_SIZE[0]
                    keypoints[s, :, 1] = keypoints[s, :, 1] * y_x_size[s, 1].item() / IMAGE_SIZE[1] 
                    mre_value = get_MRE(keypoints[s, :, :], labels_coord[s, :, :])
                    Mean_Radial_Error[click] += mre_value

    for i in range(hint_times+1):
        print("Mean Radial Error: {}".format(Mean_Radial_Error[i] / sample_count))

    # Program Ended
    print(f"\n>> End Program --- {datetime.datetime.now()} \n")
