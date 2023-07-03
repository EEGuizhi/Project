import time
import yaml
import copy
import random
import numpy as np
from munch import Munch

import torch
import torch.nn as nn
from torchvision.io import read_image
from torchvision import transforms

from model.model import IKEM
from tools.dataset import SpineDataset
from tools.heatmap_maker import HeatmapMaker
from tools.loss import LossManager


FILE_PATH = ""
IMAGE_ROOT = ""
CONFIG_PATH = ".\config\config.yaml"
CHECKPOINT_PATH = None


class Trainer(object):
    def __init__(self, model, metric_manager):
        self.model = model

        self.metric_manager = metric_manager
        self.best_epoch = None
        self.best_param = None
        self.best_metric = self.metric_manager.init_best_metric()

        self.patience = 0

    def train(self, save_manager, train_loader, val_loader, optimizer, writer=None):
        for epoch in range(1, save_manager.config.Train.epoch+1):
            start_time = time.time()

            #train
            self.model.train()
            train_loss = 0
            for i, batch in enumerate(train_loader):
                batch.detecting = False
                batch.is_training = True
                out, batch = self.forward_batch(batch)
                optimizer.update_model(out.loss)
                train_loss += out.loss.item()

                if writer is not None:
                    writer.write_loss(out.loss.item(), 'train')
                if i % 50 == 0:
                    save_manager.write_log('iter [{}/{}] loss [{:.6f}]'.format(i, len(train_loader), out.loss.item()))

            train_loss = train_loss / len(train_loader)
            train_metric = self.metric_manager.average_running_metric()

            if writer is not None:
                writer.write_metric(train_metric, 'train')

            #val
            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for i, batch in enumerate(val_loader):
                    batch.detecting = False
                    batch.is_training = True
                    out, batch = self.forward_batch(batch)
                    val_loss += out.loss.item()

                    if writer is not None:
                        writer.write_loss(out.loss.item(), 'val')
                        if epoch % 20 == 0 and i == 0:
                            writer.plot_image_heatmap(image=batch.input_image.cpu(), pred_heatmap=out.pred.heatmap.cpu(), label_heatmap=batch.label.heatmap.cpu(), epoch=epoch)
                val_metric = self.metric_manager.average_running_metric()
                val_loss = val_loss / len(val_loader)

                optimizer.scheduler_step(val_metric[save_manager.config.Train.decision_metric])


                if writer is not None:
                    writer.write_metric(val_metric, 'val')
                    if epoch % 5 == 0:
                        writer.plot_model_param_histogram(self.model, epoch)

            save_manager.write_log('Epoch [{}/{}] train loss [{:.6f}] train {} [{:.6f}] val loss [{:.6f}] val {} [{:.6f}] Epoch time [{:.1f}min]'.format(
                                    epoch,
                                    save_manager.config.Train.epoch,
                                    train_loss,
                                    save_manager.config.Train.decision_metric,
                                    train_metric[save_manager.config.Train.decision_metric],
                                    val_loss,
                                    save_manager.config.Train.decision_metric,
                                    val_metric[save_manager.config.Train.decision_metric],
                                    (time.time()-start_time)/60))
            print('version: {}'.format(save_manager.config.version))
            if self.metric_manager.is_new_best(old=self.best_metric, new=val_metric):
                self.patience = 0
                self.best_epoch = epoch
                self.best_metric = val_metric
                save_manager.save_model(self.best_epoch, self.model.state_dict(), self.best_metric)
                save_manager.write_log('Model saved after Epoch {}'.format(epoch), 4)
            else :
                self.patience += 1
                if self.patience > save_manager.config.Train.patience:
                    save_manager.write_log('Training Early Stopped after Epoch {}'.format(epoch), 16)
                    break


    def forward_batch(self, batch, metric_flag=False, average_flag=True, metric_manager=None, return_post_processing_pred=False):
        out, batch = self.model(batch)
        with torch.no_grad():
            if metric_manager is None:
                self.metric_manager.measure_metric(out.pred, batch.label, batch.pspace, metric_flag, average_flag)
            else:
                #post processing
                post_processing_pred = copy.deepcopy(out.pred)
                post_processing_pred.sargmax_coord = post_processing_pred.sargmax_coord.detach()
                post_processing_pred.heatmap = post_processing_pred.heatmap.detach()
                for i in range(len(batch.hint.index)): #for 문이 batch에 대해서 도는중 i번째 item
                    if batch.hint.index[i] is not None:
                        post_processing_pred.sargmax_coord[i, batch.hint.index[i]] = batch.label.coord[i, batch.hint.index[i]].detach().to(post_processing_pred.sargmax_coord.device)
                        post_processing_pred.heatmap[i, batch.hint.index[i]] = batch.label.heatmap[i, batch.hint.index[i]].detach().to(post_processing_pred.heatmap.device)

                metric_manager.measure_metric(post_processing_pred, copy.deepcopy(batch.label), batch.pspace, metric_flag=True, average_flag=False)
        if return_post_processing_pred:
            return out, batch, post_processing_pred
        else:
            return out, batch

    def find_worst_pred_index(self, previous_hint_index, metric_managers, save_manager, n_hint):
        batch_metric_value = metric_managers[n_hint].running_metric[save_manager.config.Train.decision_metric][-1]
        if len(batch_metric_value.shape) == 3:
            batch_metric_value = batch_metric_value.mean(-1)  # (batch, 13)

        tmp_metric = batch_metric_value.clone().detach()
        if metric_managers[n_hint].minimize_metric:
            for j, idx in enumerate(previous_hint_index):
                if idx is not None:
                    tmp_metric[j, idx] = -1000
            worst_index = tmp_metric.argmax(-1, keepdim=True)
        else:
            for j, idx in enumerate(previous_hint_index):
                if idx is not None:
                    tmp_metric[j, idx] = 1000
            worst_index = tmp_metric.argmin(-1, keepdim=True)
        return worst_index





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
        transforms.ToPILImage(mode='L'),  # grayscale
        transforms.Resize((512, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # to -1 ~ 1
    ])
    test_transform = transforms.Compose([
        transforms.ToPILImage(mode='L'),  # grayscale
        transforms.Resize((512, 256)),
        transforms.ToTensor(),
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
        print(">> training..")
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = lossManager(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i == 0:
                print(f"Epoch：{epoch} | Loss：{round(loss.item(), 3)}")

        print(">> validation..")
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = lossManager(outputs, labels)

        save_model("check_point.pth", epoch, model, optimizer)


    # Program Ended
    print("\n>> End Program --- {} \n".format(time.time()))
