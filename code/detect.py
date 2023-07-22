import os
import time
import yaml
import random
import numpy as np
from munch import Munch

import cv2
import torch

from model.model import IKEM
from tools.heatmap_maker import HeatmapMaker


INPUT_IMAGE_PATH = ""
CHECKPOINT_PATH = ""
HINT_TIMES = 10

CONFIG_PATH = "./config/config.yaml"
IMAGE_SIZE = (512, 256)
NUM_OF_KEYPOINTS = 68


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


def show_pred_image(gray_image:np.ndarray, coords:torch.Tensor, click:int):
    image = np.stack([gray_image, gray_image, gray_image], axis=-1)
    coords = coords.tolist()
    for coord in coords:
        cv2.circle(image, (int(coord[1]), int(coord[0])), 3, (255, 0, 0), -1)
    cv2.imwrite("Pred_image_{}.jpg".format(click), image)


if __name__ == '__main__':
    # Program Start
    print(f"\n>> Start Program --- {time.time()} \n")

    # Load config (yaml file)
    print("Loading Configuration..")
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    config = Munch.fromDict(config)

    # Basic settings
    set_seed(42)
    print("Using device: {}".format("cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize
    print("Initialize model...")
    model = IKEM(pretrained_model_path=None).to(device)
    heatmapMaker = HeatmapMaker(config)

    if os.path.exists(CHECKPOINT_PATH):
        print("Loading model parameters...")
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            model_param = checkpoint["model"]
            model.load_state_dict(model_param)
        except:
            print("Loading model parameters failed")
            exit()
    else:
        print(f"There is no file with file path = '{CHECKPOINT_PATH}'")
        exit()

    # Read input image
    orig_image = cv2.imread(INPUT_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    image_shape = orig_image.shape
    image = cv2.resize(orig_image, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    image = np.stack([image, image, image], axis=0)
    image = torch.tensor(image, dtype=torch.float).unsqueeze(dim=0).to(device)
    image = image / 255.0 * 2 - 1  # 0~255 to -1~1

    # Init other inputs
    hint_heatmap = torch.zeros(1, NUM_OF_KEYPOINTS, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(device)
    prev_pred = torch.zeros_like(hint_heatmap)

    # Detecting
    model.eval()
    for click in range(HINT_TIMES):
        # Model forward
        outputs, aux_out = model(hint_heatmap, prev_pred, image)

        # Plot keypoints
        keypoints = heatmapMaker.heatmap2sargmax_coord(outputs.detach().sigmoid())[0]
        keypoints[:, 0] = keypoints[:, 0] * image_shape[0] / IMAGE_SIZE[0]
        keypoints[:, 1] = keypoints[:, 1] * image_shape[1] / IMAGE_SIZE[1]
        show_pred_image(orig_image, keypoints, click)

        # User interaction
        index = int(input("\nPlease input the index of keypoints you want to fix："))
        nums = input("Please input the new coord y, x：").split(',')[0:2]

        coord = [int(num) for num in nums]
        coord = torch.tensor(coord, dtype=torch.float).unsqueeze(dim=0).unsqueeze(dim=0)
        coord[0, 0, 0] = coord[0, 0, 0] * IMAGE_SIZE[0] / image_shape[0]
        coord[0, 0, 1] = coord[0, 0, 1] * IMAGE_SIZE[1] / image_shape[1]

        # Inputs update
        prev_pred = outputs.detach().sigmoid()
        hint_heatmap[0, index] = heatmapMaker.coord2heatmap(coord)[0, 0]

    # Program Ended
    print(f"\n>> End Program --- {time.time()} \n")
