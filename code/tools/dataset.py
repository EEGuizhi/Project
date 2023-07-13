import os
import json
import numpy as np

import torch
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode


class SpineDataset(torch.utils.data.Dataset):
    def __init__(self, num_of_keypoints:int, data_file_path:str, img_root:str, transform=None, set:str="train"):
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
        self.num_of_keypoints = num_of_keypoints
        self.root = img_root
        self.transform = transform
        self.set = set

        with open(data_file_path) as f:
            self.data = json.load(f)
        for item in self.data:
            if item["set"] == set:
                self.labels.append(item["label"])
                self.images.append(os.path.join(img_root, item["image_path"]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # label (ground truth)
        label = torch.Tensor(self.labels[index])

        # image (input)
        img_path = self.images[index]
        image = read_image(img_path, mode=ImageReadMode.GRAY)
        if self.transform is not None:
            image = self.transform(image)

        # generate random hint index
        hint_indexes = torch.from_numpy(
            np.random.choice(a=self.num_of_keypoints, size=self.num_of_keypoints, replace=False)  # (不會重複)
        )

        return image, label, hint_indexes

def custom_collate_fn(samples):
    # 承接__getitem__()的東西 打包成batch
    images = torch.stack([s[0] for s in samples])
    labels = torch.stack([s[1] for s in samples])
    hint_indexes = torch.stack([s[2] for s in samples])
    return images, labels, hint_indexes
