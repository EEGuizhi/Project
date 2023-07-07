import os
import cv2
import json
import math
import numpy as np
import pandas as pd


SOURCE_PATH = "D:/python/interactive_keypoint_estimation/code/data/dataset16/boostnet_labeldata"  # dataset路徑
TARGET_PATH = "./code/dataset/"  # 存放路徑
VALIDATION_SET_SIZE = 128


def make_data(root:str, image_paths:str, label_coords:list, split:str):
    """
    打包成dictionary的格式, 方便後面處理和儲存成json檔
    """
    print(f"Making {split} data..")
    data = []
    for idx in range(len(image_paths)):
        item = {
            "image_path": image_paths[idx],
            "coners": label_coords[idx],
            "centers": None,
            "col_raw_size": None,
            "set": split
        }

        centers = []
        img_size = cv2.imread(os.path.join(root, image_paths[idx]), cv2.IMREAD_GRAYSCALE).shape  # (col_size, row_size)
        coords = np.array(label_coords[idx])
        for i in range(coords.shape[0]//4):
            centers.append([
                round(coords[i*4 : i*4 + 4, 0].mean(), 5),
                round(coords[i*4 : i*4 + 4, 1].mean(), 5)
            ])
        item["col_raw_size"] = img_size
        item["centers"] = centers

        data.append(item)
    return data


if __name__ == "__main__":
    print(">> Start Program")

    # Read labels
    base_label_path = os.path.join(SOURCE_PATH, "labels")  # dataset中存放labels的路徑
    train_image_filenames = pd.read_csv(os.path.join(base_label_path, "training", "filenames.csv"), header=None).values[:, 0].tolist()
    test_image_filenames = pd.read_csv(os.path.join(base_label_path, "test", "filenames.csv"), header=None).values[:, 0].tolist()
    train_label_landmarks = pd.read_csv(os.path.join(base_label_path, "training", "landmarks.csv"), header=None).values
    test_label_landmarks = pd.read_csv(os.path.join(base_label_path, "test", "landmarks.csv"), header=None).values

    train_image_paths = ["data/training/"+path for path in train_image_filenames]
    test_image_paths = ["data/test/"+path for path in test_image_filenames]

    train_labels = []
    test_labels = []
    for idx in range(train_label_landmarks.shape[0]):
        keypoints = []
        for i in range(train_label_landmarks.shape[1]//2):
            keypoints.append(train_label_landmarks[idx, i*2:i*2+2].tolist())
        train_labels.append(keypoints)
    for idx in range(test_label_landmarks.shape[0]):
        keypoints = []
        for i in range(test_label_landmarks.shape[1]//2):
            keypoints.append(test_label_landmarks[idx, i*2:i*2+2].tolist())
        test_labels.append(keypoints)
    del train_image_filenames, test_image_filenames, train_label_landmarks, test_label_landmarks

    # Generate validation dataset
    val_idx = sorted(np.random.choice(len(train_labels), size=VALIDATION_SET_SIZE, replace=False))  # 隨機選擇index
    train_idx = [i for i in range(len(train_labels)) if i not in val_idx]

    val_image_paths = np.array(train_image_paths)[val_idx].tolist()
    train_image_paths = np.array(train_image_paths)[train_idx].tolist()
    val_labels = np.array(train_labels)[val_idx].tolist()
    train_labels = np.array(train_labels)[train_idx].tolist()

    # Make json files data
    train_data = make_data(SOURCE_PATH, train_image_paths, train_labels, "train")
    val_data = make_data(SOURCE_PATH, val_image_paths, val_labels, "val")
    test_data = make_data(SOURCE_PATH, test_image_paths, test_labels, "test")

    all_data = train_data
    for item in val_data: all_data.append(item)
    for item in test_data: all_data.append(item)

    # Save data
    if not os.path.exists(TARGET_PATH):
        os.makedirs(TARGET_PATH)
    with open(os.path.join(TARGET_PATH, "train_data.json"), 'w') as f:
        json.dump(train_data, f)
    with open(os.path.join(TARGET_PATH, "val_data.json"), 'w') as f:
        json.dump(val_data, f)
    with open(os.path.join(TARGET_PATH, "test_data.json"), 'w') as f:
        json.dump(test_data, f)
    with open(os.path.join(TARGET_PATH, "all_data.json"), 'w') as f:
        json.dump(all_data, f)

    print(">> End Program")
