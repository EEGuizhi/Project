import os
import cv2
import json
import numpy as np

import torch


class SpineDataset(torch.utils.data.Dataset):
    def __init__(self, image_size:tuple, num_of_keypoints:int, data_file_path:str, img_root:str, transform=None, set:str="train"):
        """
        Parameters:
        ===
            data_file_path: .json file has image paths and labels data.
            img_root: the path of folder containing images.
            transform: transformation for this dataset.
            set: "train" or "val" or "test". Defaults to "train".
        """
        self.labels = []
        self.images = []
        self.y_x_size = []
        self.image_size = image_size
        self.num_of_keypoints = num_of_keypoints
        self.root = img_root
        self.set = set
        self.transform = transform

        with open(data_file_path) as f:
            self.data = json.load(f)
        for item in self.data:
            if item["set"] == set:
                corners = np.array(item["corners"]) * np.array(item["y_x_size"])
                self.labels.append(corners.astype(np.int32).tolist())
                self.images.append(os.path.join(img_root, item["image_path"]))
                self.y_x_size.append(item["y_x_size"])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # image (input) & label (ground truth)
        img_path = self.images[index]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = np.stack([image, image, image], axis=-1)
        label = self.labels[index]

        transformed = self.transform(image=image, keypoints=label)
        image = transformed["image"]
        label = torch.tensor(transformed["keypoints"])

        # np array to tensor
        image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)
        image = image / 255.0 * 2 - 1  # 0~255 to -1~1

        # generate random hint index (not used)
        hint_indexes = torch.from_numpy(
            np.random.choice(a=self.num_of_keypoints, size=self.num_of_keypoints, replace=False)  # (不會重複)
        )

        return image, label, hint_indexes, self.y_x_size[index]

def custom_collate_fn(samples):
    # 承接__getitem__()的東西 打包成batch
    images = torch.stack([s[0] for s in samples])
    labels = torch.stack([s[1] for s in samples])
    hint_indexes = torch.stack([s[2] for s in samples])
    y_x_size = torch.stack([s[3] for s in samples])
    return images, labels, hint_indexes, y_x_size
