import os
import copy
import json
import numpy as np
from munch import Munch

import torch
from torchvision.io import read_image

class SpineDataset(torch.utils.data.Dataset):
    def __init__(self, data_file_path:str, img_root:str, transform=None, set:str="train"):
        """
        Args:
            data_file_path: .json file has image paths and labels data.
            img_root: the path of folder containing images.
            transform: transformation for this dataset.
            set: "train" or "val" or "test". Defaults to "train".
        """
        self.root = img_root
        self.transform = transform
        self.labels = []
        self.inputs = []

        with open(data_file_path) as f:
            self.data = json.load(f)
        for item in self.data:
            if item["set"] == set:
                self.labels.append(item["label"])
                self.inputs.append(item["image_path"])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # indexing
        item = self.data[index]

        # image load
        img_path = os.path.join(self.config.PATH.DATA.IMAGE, item[self.config.DICT_KEY.IMAGE])
        
        img, row, column = self.loadimage(img_path, item[self.config.DICT_KEY.RAW_SIZE])

        # pixel spacing
        pspace_list = item[self.config.DICT_KEY.PSPACE] # row, column
        raw_size_and_pspace = torch.tensor([row, column] + pspace_list)  # 串聯

        # points load (13,2) (column, row)==(xy)
        coords = copy.deepcopy(item[self.config.DICT_KEY.POINTS])
        coords.append([1.0,1.0])

        transformed = self.transformer(image=img, keypoints=coords)
        img, coords = transformed["image"], transformed["keypoints"]
        additional = torch.tensor([])

        coords = np.array(coords)

        # np array to tensor (800, 640)=(row, col)
        img = torch.tensor(img, dtype=torch.float)
        img = img.permute(2, 0, 1)
        img /= 255.0  # 0~255 to 0~1
        img = img * 2 - 1  # 0~1 to -1~1

        coords = torch.tensor(copy.deepcopy(coords[:, ::-1]), dtype=torch.float)
        morph_loss_mask = (coords[-1] == torch.tensor([1.0, 1.0], dtype=torch.float)).all()
        coords = coords[:-1]

        # hint
        if self.split == 'train':
            # random hint
            num_hint = np.random.choice(range(self.config.Dataset.num_keypoint ), size=None, p=self.config.Hint.num_dist)  # config.Hint.num_dist在util.py的config_name2value()中
            hint_indices = np.random.choice(range(self.config.Dataset.num_keypoint ), size=num_hint, replace=False) #[1,2,3]
        else:
            hint_indices = None

        return img_path, img, raw_size_and_pspace, hint_indices, coords, additional, index, morph_loss_mask
