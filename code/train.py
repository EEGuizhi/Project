import time
import yaml
import copy
import random
import numpy as np
from munch import Munch

import torch
import torch.nn as nn
from torchvision import transforms

from model.model import IKEM
from tools.dataset import SpineDataset
from tools.heatmap_maker import HeatmapMaker
from tools.loss import LossManager


FILE_PATH = ""
IMAGE_ROOT = ""
CONFIG_PATH = ".\config\config.yaml"
CHECKPOINT_PATH = None

IMAGE_SIZE = (512, 256)
NUM_KEYPOINTS = 17

EPOCH = 1000
BATCH_SIZE = 32
LR = 1e-3


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

def save_model(path:str, epoch:int, model:nn.Module, optimizer:torch.optim.Optimizer):
    print("Saving model..")
    save_dict = {
        "model": model.state_dict(),
        "epoch": epoch,
        "optimizer": optimizer.state_dict()
    }
    torch.save(save_dict, path)


if __name__ == '__main__':
    # Program Start
    print("\n>> Start Program --- {} \n".format(time.time()))

    # Load config (yaml file)
    print("Loading Configuration..")
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    config = Munch.fromDict(config)

    # Settings
    set_seed(config.seed if config.seed is not None else 42)
    print("Using device: {}".format("cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),  # 0 ~ 255 to -1 ~ 1
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # to -1 ~ 1
    ])
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),  # 0 ~ 255 to -1 ~ 1
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # to -1 ~ 1
    ])
    train_set = SpineDataset(data_file_path=FILE_PATH, img_root=IMAGE_ROOT, transform=train_transform, set="train")
    val_set = SpineDataset(data_file_path=FILE_PATH, img_root=IMAGE_ROOT, transform=train_transform, set="val")
    test_set = SpineDataset(data_file_path=FILE_PATH, img_root=IMAGE_ROOT, transform=test_transform, set="test")
    train_loader = torch.utils.data.DataLoader(train_set, BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, BATCH_SIZE, shuffle=True)

    # Initialize
    print("Initialize model...")
    model = IKEM(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # https://medium.com/%E9%9B%9E%E9%9B%9E%E8%88%87%E5%85%94%E5%85%94%E7%9A%84%E5%B7%A5%E7%A8%8B%E4%B8%96%E7%95%8C/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92ml-note-sgd-momentum-adagrad-adam-optimizer-f20568c968db
    heatmapMaker = HeatmapMaker(config)
    lossManager = LossManager(use_coord_loss=True, heatmap_maker=heatmapMaker)

    if CHECKPOINT_PATH is not None:
        print("Loading model parameters...")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model_param = checkpoint["model"]
        model.load_state_dict(model_param)
        try:
            start_epoch = checkpoint["epoch"]
            optimizer_param = checkpoint["optimizer"]
            optimizer = torch.optim.Optimizer.load_state_dict(optimizer_param)
        except:
            start_epoch = 0
            optimizer = None
        del model_param, optimizer_param, checkpoint
    else:
        start_epoch = 0

    # Calculate the number of model parameters
    n_params = 0
    for k, v in model.named_parameters():  # 遍歷model每一層, k是名稱, v是參數值
        n_params += v.reshape(-1).shape[0]  # v是一個tensor, reshape(-1)表示將v展平; shape[0]表示v展平後的元素個數
    print('Number of model parameters: {}'.format(n_params))

    # Training
    for epoch in range(start_epoch, EPOCH+1):
        print(f"\n>> Epoch:{epoch}")
        model.train()
        train_loss = 0
        for i, (inputs, labels, hint_indexes) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            if hint_indexes is not None:
                with torch.no_grad():
                    model.eval()
                    num_iters = np.random.randint(0, self.max_iter)
                    pred_heatmap = torch.zeros_like(input_hint_heatmap)
                    for click_indx in range(num_iters):
                        # prediction
                        pred_logit, aux_pred_logit = self.forward_model(input_image, input_hint_heatmap, pred_heatmap)
                        pred_heatmap = pred_logit.sigmoid()

                        # hint update (training 때는 hint를 줘가면서 update하는 과정을 거침)
                        batch = self.get_next_points(batch, pred_heatmap)
                        for i in range(batch.label.heatmap.shape[0]):
                            if batch.hint.index[i] is not None:
                                batch.hint.heatmap[i, batch.hint.index[i]] = batch.label.heatmap[i, batch.hint.index[i]]
                    model.train()
                outputs = model(inputs)
            else:
                hint_heatmap = torch.zeros(NUM_KEYPOINTS, IMAGE_SIZE[0], IMAGE_SIZE[1])
                prev_pred = torch.zeros_like(hint_heatmap)
                hint_heatmap
                outputs = model(hint_heatmap, prev_pred, inputs)

            loss = lossManager(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() / len(train_loader)
        print(f"Training Loss：{round(train_loss, 3)}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (inputs, labels, hint_indexes) in enumerate(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = lossManager(outputs, labels)

                val_loss += loss.item() / len(val_loader)
        print(f"Validation Loss：{round(val_loss, 3)}")

        save_model("checkpoint_{}.pth".format(epoch//50), epoch, model, optimizer)

    # Program Ended
    print("\n>> End Program --- {} \n".format(time.time()))
