import os
import random
import datetime
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms

import cv2
import json
import torch
import numpy as np

from model.model import IKEM
from model.unet_IKEM import UNet_IKEM
from tools.heatmap_maker import HeatmapMaker


USE_GUI = False
INPUT_IMAGE = "sunhl-1th-01-Mar-2017-311 M AP.jpg"  # an image in the test folder
AUTO_REVISE_LABEL_ROOT = "D:/GitHub/EEGuizhi/Project/code/dataset/all_data.json"  # 限有標籤的資料


# Model
MODELS = ["HRNetOCR_IKEM", "UNet_IKEM"]
USE_MODEL = 1


INPUT_IMAGE_PATH = "D:/python/interactive_keypoint_estimation/code/data/dataset16/boostnet_labeldata/data/test/"+INPUT_IMAGE
HIST_MATCH_IMAGE = ""
CHECKPOINT_PATH = "models_saved/UNet_IKEM_12_21.pth"
HINT_TIMES = 10

IMAGE_SIZE = (512, 256)
HEATMAP_STD = 7.5
NUM_OF_KEYPOINTS = 68

COLOR_CODE = [
    (255,  0,  0), (  0,255,  0), (  0,  0,255), (255,255,  0), (  0,255,255), ( 46,139, 87),
    (255,  0,  0), (  0,  0,205), (205,133, 63), (  0,255,  0), (  0,  0,255), (  0,  0,128),
    (  0,139,139), ( 46,139, 87), (255,255,  0), (106, 90,205), (  0,255,255)
]


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


def show_pred_image(gray_image:np.ndarray, coords:torch.Tensor, click:int, save_folder:str=''):
    image = np.stack([gray_image, gray_image, gray_image], axis=-1)
    coords = coords.tolist()
    radius = 10
    i = 0
    for coord in coords:
        if i%4 == 0:
            cv2.putText(image, f"{i//4 + 1}", (int(coord[1])+radius, int(coord[0])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_CODE[i//4], 2)
        cv2.circle(image, (int(coord[1]), int(coord[0])), radius, COLOR_CODE[i//4], -1)
        i += 1
    if save_folder != '': save_folder += '/'
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_folder+"Pred_image_{}.jpg".format(click), image)
    return image if USE_GUI else None


fix_coord = None
true_coord = None
def onclick(event):
    global fix_coord, true_coord
    x, y = event.xdata, event.ydata
    print(f"Click x, y = {x}, {y} on picture")
    if fix_coord is None:
        fix_coord = torch.tensor([y, x])
        plt.plot(x, y, 'rx')  # red X mark
    else:
        true_coord = [y, x]
        plt.plot(x, y, 'gX')  # green X mark
    plt.draw()


def keypoints_perc2coord(coords:np.ndarray, img_size:tuple):
    real_coords = np.empty_like(coords)
    real_coords[:, 0] = coords[:, 0] * img_size[0]
    real_coords[:, 1] = coords[:, 1] * img_size[1]
    return real_coords.astype(int).tolist()


if __name__ == '__main__':
    # Program Start
    print(f"\n>> Start Program --- {datetime.datetime.now()} \n")

    # Auto revise
    corner_coords = None
    center_coords = None
    if AUTO_REVISE_LABEL_ROOT != "" and AUTO_REVISE_LABEL_ROOT != None:
        with open(AUTO_REVISE_LABEL_ROOT, 'r') as f:
            label_data = json.loads(f.read())

        spec_data = INPUT_IMAGE_PATH.split("/")[-1]

        # Show specific images first
        for data in label_data:
            if spec_data in data["image_path"]:
                corner_coords = torch.tensor(keypoints_perc2coord(np.array(data["corners"]), data["y_x_size"]))
                center_coords = torch.tensor(keypoints_perc2coord(np.array(data["centers"]), data["y_x_size"]))

    # Basic settings
    set_seed(42)
    print("Using device: {}".format("cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize
    print("Initialize model...")
    if MODELS[USE_MODEL] == "HRNetOCR_IKEM":
        model = IKEM(pretrained_model_path=None).to(device)
    elif MODELS[USE_MODEL] == "UNet_IKEM":
        model = UNet_IKEM().to(device)
    heatmapMaker = HeatmapMaker(IMAGE_SIZE, HEATMAP_STD)

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

    if os.path.exists(HIST_MATCH_IMAGE):
        # Match image
        target_image = cv2.imread(HIST_MATCH_IMAGE, cv2.IMREAD_GRAYSCALE)
        target_image = cv2.resize(target_image, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
        matched_image = match_histograms(
            np.stack([image, image, image], axis=-1),
            np.stack([target_image, target_image, target_image], axis=-1)
        )
        cv2.imwrite("matched_input_image.jpg", matched_image)
        image = torch.tensor(matched_image, dtype=torch.float, device=device).permute(2, 0, 1).unsqueeze(dim=0)
    else:
        image = torch.tensor(image, dtype=torch.float, device=device)
        image = torch.stack([image, image, image]).unsqueeze(dim=0)
    image = image / 255.0 * 2 - 1  # 0~255 to -1~1

    # Init other inputs
    hint_heatmap = torch.zeros(1, NUM_OF_KEYPOINTS, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(device)
    prev_pred = torch.zeros_like(hint_heatmap)

    # Detecting
    i = 0
    while os.path.exists("predictions_{}".format(i)):
        i += 1
    target_folder = "predictions_{}".format(i)
    os.makedirs(target_folder)
    with torch.no_grad():
        manual_revision = []
        model.eval()
        for click in range(HINT_TIMES+1):
            print(f">> Detection start time: {datetime.datetime.now()}")
            # Model forward
            if MODELS[USE_MODEL] == "HRNetOCR_IKEM":
                outputs, aux_out = model(hint_heatmap, prev_pred, image)
            elif MODELS[USE_MODEL] == "UNet_IKEM":
                outputs = model(hint_heatmap, prev_pred, image)
            print(f">> Detection finish time: {datetime.datetime.now()}")

            # Plot keypoints
            plot_heatmap = outputs.cpu().detach().sigmoid()[0][10]
            plot_heatmap = plot_heatmap.numpy()
            plot_heatmap = plot_heatmap * 255
            print(f"plot_heatmap.max() = {plot_heatmap.max()}")
            cv2.imwrite("heatmap_of_keypoint_10th.jpg", plot_heatmap)
            keypoints = heatmapMaker.heatmap2expected_coord(outputs.detach().sigmoid())[0]
            keypoints[:, 0] = keypoints[:, 0] * image_shape[0] / IMAGE_SIZE[0]
            keypoints[:, 1] = keypoints[:, 1] * image_shape[1] / IMAGE_SIZE[1]
            for item in manual_revision:
                keypoints[item["index"], 0], keypoints[item["index"], 1] = item["coord"][0], item["coord"][1]
            pred_image = show_pred_image(orig_image, keypoints, click, target_folder)

            if click < HINT_TIMES:
                if USE_GUI:
                    fig = plt.figure("Pred image of click {}".format(click))
                    plt.imshow(pred_image)
                    plt.xticks([]), plt.yticks([])

                    print("\nPlease click the point you would like to correct First")
                    print("then click the new coord of that point on picture.\n")
                    fig.canvas.mpl_connect('button_press_event', onclick)
                    plt.show()

                    # Find closest point index
                    fix_coord = fix_coord.repeat(keypoints.shape[0], 1)
                    keypoints = keypoints.cpu() - fix_coord
                    index = torch.argmin(torch.sum(torch.pow(keypoints, 2), dim=-1)).item()
                    coord = true_coord

                    fix_coord = None
                else:
                    if AUTO_REVISE_LABEL_ROOT:
                        keypoints = keypoints.cpu() - corner_coords
                        index = torch.argmax(torch.sum(torch.pow(keypoints, 2), dim=-1)).item()
                        coord = corner_coords[index]
                    else:
                        # User interaction
                        index = int(input("\nPlease input the index of keypoints you want to fix："))
                        nums = input("Please input the new coord y, x：").split(',')[0:2]
                        coord = [int(num) for num in nums]

                manual_revision.append({
                    "index": index,
                    "coord": coord
                })
                coord = torch.tensor(coord, dtype=torch.float).unsqueeze(dim=0).unsqueeze(dim=0)
                coord[0, 0, 0] = coord[0, 0, 0] * IMAGE_SIZE[0] / image_shape[0]
                coord[0, 0, 1] = coord[0, 0, 1] * IMAGE_SIZE[1] / image_shape[1]

                # Inputs update
                prev_pred = outputs.detach().sigmoid()
                hint_heatmap[0, index] = heatmapMaker.coord2heatmap(coord)[0, 0]

    # Program Ended
    print(f"\n>> End Program --- {datetime.datetime.now()} \n")
