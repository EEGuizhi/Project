import os
import cv2
import json
import numpy as np


SHOW_INTERVAL = 0  # (ms) (= 0 會等到按鍵按下)
DATASET_ROOT = "D:/python/interactive_keypoint_estimation/code/data/dataset16/boostnet_labeldata"
LABEL_FILE_PATH = "code/dataset/all_data.json"
KEYPOINTS_PERC2COORD = True

SHOW_FIRST_LIST = [
    "sunhl-1th-02-Jan-2017-162 B AP.jpg",
    "sunhl-1th-03-Jan-2017-163 B AP.jpg",
    "sunhl-1th-05-Jan-2017-167 A AP.jpg",
    "sunhl-1th-06-Jan-2017-184 A AP.jpg",
    "sunhl-1th-06-Jan-2017-187 A AP.jpg",
    "sunhl-1th-06-Jan-2017-188 B AP.jpg"
]


def show_image(data:dict):
    # Get image file path
    image_path = data["image_path"]

    # Read image
    img = cv2.imread(os.path.join(DATASET_ROOT, image_path))

    # Get keypoints coord
    if KEYPOINTS_PERC2COORD:
        corner_coords = keypoints_perc2coord(np.array(data["corners"]), data["x_y_size"])
        center_coords = keypoints_perc2coord(np.array(data["centers"]), data["x_y_size"])
    else:
        corner_coords = data["corners"]
        center_coords = data["centers"]

    # Plot
    kpt_img = plot_keypoints(img=img, corners=corner_coords, centers=center_coords)
    print(f"Showing image: {image_path}        \r", end='')
    cv2.imshow("{}".format(image_path), kpt_img)
    cv2.imwrite("keypoints_plot_image.jpg", kpt_img)
    cv2.waitKey(SHOW_INTERVAL)
    cv2.destroyAllWindows()


def plot_keypoints(img:np.ndarray, corners:list, centers:list):
    for coord in corners:
        cv2.circle(img, (coord[0], coord[1]), 3, (255, 0, 0), -1)
    for coord in centers:
        cv2.circle(img, (coord[0], coord[1]), 3, (0, 0, 255), -1)
    return img


def keypoints_perc2coord(coords:np.ndarray, img_size:tuple):
    real_coords = np.empty_like(coords)
    real_coords[:, 0] = coords[:, 0] * img_size[0]
    real_coords[:, 1] = coords[:, 1] * img_size[1]
    return real_coords.astype(int).tolist()


if __name__ == "__main__":
    # Read data
    with open(LABEL_FILE_PATH, 'r') as f:
        label_data = json.loads(f.read())

    # Show specific images first
    print("\nShowing specific images first:")
    for spec_data in SHOW_FIRST_LIST:
        if spec_data != "" or spec_data != None:
            for data in label_data:
                if spec_data in data["image_path"]: show_image(data)

    # Show images
    print("\nShowing all images:")
    for data in label_data:
        show_image(data)
