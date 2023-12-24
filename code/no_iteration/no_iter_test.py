import os
import datetime

import torch
import albumentations as A  # keypoints_augmentation：https://albumentations.ai/docs/getting_started/keypoints_augmentation

from model.hrnet_ocr import hrnet_ocr
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
    model = hrnet_ocr(pretrained_model_path=None).to(device)
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
    Model_MRE = 0
    with torch.no_grad():
        model.eval()
        for i, (images, labels, hint_indexes, y_x_size) in enumerate(test_loader):
            # Init
            sample_count += images.shape[0]
            images = images.to(device)
            labels = labels.to(device)
            y_x_size = y_x_size.to(device)
            image_size = torch.tensor(IMAGE_SIZE, device=device)

            origSize_labels = torch.zeros_like(labels)
            for s in range(labels.shape[0]):  # get labels coord in original size
                origSize_labels[s] = labels[s] * y_x_size[s] / image_size

            outputs, aux_out = model(images)
            keypoints = heatmapMaker.heatmap2expected_coord(outputs.sigmoid())

            # Get MRE
            for s in range(images.shape[0]):
                # Scale to original size
                keypoints[s] = keypoints[s] * y_x_size[s] / image_size

                # Calc MRE
                Model_MRE += get_MRE(keypoints[s], origSize_labels[s])

    # Outputs
    print(f"Mean Radial Error: {Model_MRE / sample_count}")

    # Program Ended
    print(f"\n>> End Program --- {datetime.datetime.now()} \n")
