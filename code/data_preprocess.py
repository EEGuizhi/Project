import os
import cv2
import json
import scipy.io
import numpy as np
import pandas as pd


SOURCE_PATH = "D:/python/interactive_keypoint_estimation/code/data/dataset16/boostnet_labeldata"  # dataset路徑
TARGET_PATH = "./code/dataset/"  # 存放路徑
VALIDATION_SET_SIZE = 128
FROM_CSVFILE = False  # read landmarks from landmarks.csv (讀出來的座標會是相對座標(百分比))


def make_data(root:str, image_paths:str, label_coords:list, split:str):
    """
    打包成dictionary的格式, 方便後面處理和儲存成json檔
    """
    print(f"Making {split} data.. ({len(image_paths)} samples)")
    data = []
    for idx in range(len(image_paths)):
        item = {
            "image_path": image_paths[idx],
            "corners": label_coords[idx],
            "centers": None,
            "x_y_size": None,
            "set": split
        }

        # check data type
        if type(item["corners"]) != list:
            item["corners"] = item["corners"].tolist()

        # data
        centers = []
        y_size, x_size = cv2.imread(os.path.join(root, image_paths[idx]), cv2.IMREAD_GRAYSCALE).shape  # (col_size=y, row_size=x)
        coords = np.array(label_coords[idx])
        for i in range(coords.shape[0]//4):
            centers.append([
                round(coords[i*4 : i*4 + 4, 0].mean(), 5 if FROM_CSVFILE else None),
                round(coords[i*4 : i*4 + 4, 1].mean(), 5 if FROM_CSVFILE else None)
            ])
        item["x_y_size"] = (x_size, y_size)  # (x_size, y_size)
        item["centers"] = centers

        data.append(item)
    return data


def check_error_data(root:str, image_paths:list, labels:list, split:str):
    """
    移除超出圖片範圍的 keypoints (labels)
    """
    print(f"Checking Error data in {split} set..")
    remove_list = []
    coords = np.array(labels)
    for idx, path in enumerate(image_paths):
        y_size, x_size = cv2.imread(os.path.join(root, path), cv2.IMREAD_GRAYSCALE).shape  # (col_size=y, row_size=x)
        if coords[idx, :, 0].max() > x_size:
            print(f"error: idx={idx} - exceed max x")
            remove_list.append(idx)
        elif coords[idx, :, 1].max() > y_size:
            print(f"error: idx={idx} - exceed max y")
            remove_list.append(idx)
    for i, idx in enumerate(remove_list):
        del image_paths[idx-i], labels[idx-i]
    return image_paths, labels


if __name__ == "__main__":
    print(">> Start Program")

    # Read csv files
    base_label_path = os.path.join(SOURCE_PATH, "labels")  # dataset中存放labels的路徑
    train_image_filenames = pd.read_csv(os.path.join(base_label_path, "training", "filenames.csv"), header=None).values[:, 0].tolist()
    test_image_filenames = pd.read_csv(os.path.join(base_label_path, "test", "filenames.csv"), header=None).values[:, 0].tolist()
    if FROM_CSVFILE:
        train_label_landmarks = pd.read_csv(os.path.join(base_label_path, "training", "landmarks.csv"), header=None).values
        test_label_landmarks = pd.read_csv(os.path.join(base_label_path, "test", "landmarks.csv"), header=None).values

    # Image paths & Keypoints coord
    train_image_paths = ["data/training/"+path for path in train_image_filenames]
    test_image_paths = ["data/test/"+path for path in test_image_filenames]
    train_labels = []
    test_labels = []

    if FROM_CSVFILE:
        for idx in range(train_label_landmarks.shape[0]):
            keypoints = []
            for i in range(train_label_landmarks.shape[1]//2):
                keypoints.append([
                    train_label_landmarks[idx, i],
                    train_label_landmarks[idx, train_label_landmarks.shape[1]//2 + i]
                ])
            train_labels.append(keypoints)
        for idx in range(test_label_landmarks.shape[0]):
            keypoints = []
            for i in range(test_label_landmarks.shape[1]//2):
                keypoints.append([
                    test_label_landmarks[idx, i],
                    test_label_landmarks[idx, test_label_landmarks.shape[1]//2 + i]
                ])
            test_labels.append(keypoints)
        del train_image_filenames, test_image_filenames, train_label_landmarks, test_label_landmarks

    else:
        for img_path in train_image_paths:
            label_path = img_path.replace("data", "labels")+".mat"
            label = scipy.io.loadmat(os.path.join(SOURCE_PATH, label_path))['p2']  # get keypoints coord (x, y)
            train_labels.append(label)
        for img_path in test_image_paths:
            label_path = img_path.replace("data", "labels")+".mat"
            label = scipy.io.loadmat(os.path.join(SOURCE_PATH, label_path))['p2']  # get keypoints coord (x, y)
            test_labels.append(label)

    # Generate validation dataset
    val_idx = sorted(np.random.choice(len(train_labels), size=VALIDATION_SET_SIZE, replace=False))  # 隨機選擇index
    train_idx = [i for i in range(len(train_labels)) if i not in val_idx]

    val_image_paths = np.array(train_image_paths)[val_idx].tolist()
    train_image_paths = np.array(train_image_paths)[train_idx].tolist()
    val_labels = np.array(train_labels, dtype=np.float32 if FROM_CSVFILE else np.int32)[val_idx].tolist()
    train_labels = np.array(train_labels, dtype=np.float32 if FROM_CSVFILE else np.int32)[train_idx].tolist()

    # Del Error data
    if FROM_CSVFILE is False:
        train_image_paths, train_labels = check_error_data(SOURCE_PATH, train_image_paths, train_labels, "train")
        val_image_paths, val_labels = check_error_data(SOURCE_PATH, val_image_paths, val_labels, "val")
        test_image_paths, test_labels = check_error_data(SOURCE_PATH, test_image_paths, test_labels, "test")

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
