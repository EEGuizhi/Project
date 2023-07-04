import os
import json
import math
import numpy as np

import torch
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode

class SpineDataset(torch.utils.data.Dataset):
    def __init__(self, data_file_path:str, img_root:str, transform=None, set:str="train"):
        """
        Args:
        ===
            data_file_path: .json file has image paths and labels data.
            img_root: the path of folder containing images.
            transform: transformation for this dataset.
            set: "train" or "val" or "test". Defaults to "train".
        """
        self.root = img_root
        self.transform = transform
        self.labels = []
        self.imgs = []
        self.set = set

        with open(data_file_path) as f:
            self.data = json.load(f)
        for item in self.data:
            if item["set"] == set:
                self.labels.append(item["label"])
                self.imgs.append(os.path.join(img_root, item["image_path"]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # label (ground truth)
        label = self.labels[index]
        num_of_keypoints = len(label)

        # image (input)
        img_path = self.imgs[index]
        image = read_image(img_path, mode=ImageReadMode.GRAY)
        if self.transform is not None:
            image = self.transform(image)

        # generate random hint indexes
        prob = np.array([math.pow(2, -i) for i in range(num_of_keypoints)])
        prob = prob.tolist() / prob.sum()
        if self.set == "train":
            hint_times = np.random.choice(a=num_of_keypoints, size=None, p=prob)  # 決定總共提示次數
            hint_indexes = np.random.choice(a=num_of_keypoints, size=hint_times, replace=False)  # 決定每次提示的為哪個關鍵點(不會重複)
        else:
            hint_indexes = None

        return image, label, hint_indexes
