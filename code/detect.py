import os
import json
import time
import copy
import torch
import random
import argparse
import datetime
import scipy.io
import cv2
import numpy as np
import torch.nn as nn

from PIL import Image
from pytz import timezone
from munch import Munch
from tqdm.auto import tqdm

from util import SaveManager, TensorBoardManager
from model import get_model
from dataset import get_dataloader
# from misc.metric import MetricManager
# from misc.optimizer import get_optimizer
# from misc.train import Trainer

IMG_SOURCE_PATH = "/content/code/data/dataset16/boostnet_labeldata/"
INPUT_IMAGE_PATH = "/content/code/data/dataset16/boostnet_labeldata/data/test/sunhl-1th-01-Mar-2017-310 C AP.jpg"
HINT_TIMES = 5


torch.set_num_threads(4)  # 設置在CPU上平行運算時所佔用的線程數


def parse_args():
    '''
    解析傳入的參數。詳見:https://shengyu7697.github.io/python-argparse/
    '''
    parser = argparse.ArgumentParser(description='TMI experiments')  # 創建
    arg = parser.parse_args()  # 解析
    arg.config = '_'
    arg.seed = 42
    arg.only_test_version = "ExpNum[00001]_Dataset[dataset16]_Model[RITM_SE_HRNet32]_config[spineweb_ours]_seed[42]"
    arg.save_test_prediction = True
    
    ### set seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(arg.seed)
    torch.cuda.manual_seed(arg.seed)
    torch.cuda.manual_seed_all(arg.seed)
    np.random.seed(arg.seed)
    random.seed(arg.seed)
    
    return arg


class Detector(object):
    def __init__(self, model):
        self.model = model
        # self.metric_manager = metric_manager
        self.best_epoch = None
        self.best_param = None
        # self.best_metric = self.metric_manager.init_best_metric()
        self.patience = 0

    def detect(self, save_manager, input_image, writer=None):
        # To reduce GPU usage, load params on cpu and upload model on gpu(params on gpu also need additional GPU memory)
        self.model.cpu()
        self.model.load_state_dict(self.best_param)
        self.model.to(save_manager.config.MISC.device)

        input_image = torch.tensor([input_image.tolist()], dtype=torch.float, device=save_manager.config.MISC.device)
        input_image /= 255.0
        input_image = input_image * 2 - 1

        # save_manager.write_log('Best model at epoch {} loaded'.format(self.best_epoch))
        self.model.eval()
        with torch.no_grad():
            # post_metric_managers = [MetricManager(save_manager) for _ in range(save_manager.config.Dataset.num_keypoint+1)]
            # manual_metric_managers = [MetricManager(save_manager) for _ in range(save_manager.config.Dataset.num_keypoint+1)]

            # for i, batch in enumerate(tqdm(test_loader)):
            batch = {
                "input_image": input_image,
                "is_training": False,
                "detecting": True,
                "label": None,
                "hint": {
                    "index": None,
                    "coords": None
                }
            }
            batch = Munch.fromDict(batch)

            for n_hint in range(HINT_TIMES + 1):
                ## Model forward
                out, batch, post_processing_pred = self.forward_batch(batch, return_post_processing_pred=True)
                if n_hint > 0:
                    batch.prev_heatmap = out.pred.heatmap.detach()

                print(">> post_processing_pred:\n", post_processing_pred.sargmax_coord)

                # ## Manual forward
                # if n_hint == 0:
                #     manual_pred = copy.deepcopy(out.pred)
                #     manual_pred.sargmax_coord = manual_pred.sargmax_coord
                #     manual_pred.heatmap = manual_pred.heatmap
                #     manual_hint_index = []
                # else:
                #     # Manual prediction update
                #     for k in range(manual_hint_index.shape[0]):
                #         manual_pred.sargmax_coord[k, manual_hint_index[k]] = batch.label.coord[k, manual_hint_index[k]].to(manual_pred.sargmax_coord.device)
                #         manual_pred.heatmap[k, manual_hint_index[k]] = batch.label.heatmap[k, manual_hint_index[k]].to(manual_pred.heatmap.device)
                # manual_metric_managers[n_hint].measure_metric(manual_pred, batch.label, batch.pspace, metric_flag=True, average_flag=False)

                # ============================= model hint =================================
                # worst_index = self.find_worst_pred_index(batch.hint.index, post_metric_managers, save_manager, n_hint)
                worst_index = input(">> Please input the worst index : ")
                worst_index = int(worst_index)
                coord = input(">> Please input the coords of the worst keypoint (num1,num2) : ")
                coord = [float(coord.split(',')[0]), float(coord.split(',')[1])]
                # Hint index update
                if n_hint == 0:
                    batch.hint.index = [worst_index]
                    batch.hint.coords = [coord]
                else:
                    batch.hint.index.append(worst_index)
                    batch.hint.coords.append(coord)

                # if save_manager.config.Model.use_prev_heatmap_only_for_hint_index:
                #     new_prev_heatmap = torch.zeros_like(out.pred.heatmap)
                #     for j in range(len(batch.hint.index)):
                #         new_prev_heatmap[j, batch.hint.index[j]] = out.pred.heatmap[j, batch.hint.index[j]]
                #     batch.prev_heatmap = new_prev_heatmap
                # ================================= manual hint =========================
                # worst_index = self.find_worst_pred_index(manual_hint_index, manual_metric_managers, save_manager, n_hint)
                # # Hint index update
                # if n_hint == 0:
                #     manual_hint_index = worst_index  # (batch, 1)
                # else:
                #     manual_hint_index = torch.cat((manual_hint_index, worst_index.to(manual_hint_index.device)), dim=1)  # ... (batch, max hint)

                # # Save result
                # if save_manager.config.save_test_prediction:
                #     save_manager.add_test_prediction_for_save(batch, post_processing_pred, manual_pred, n_hint, post_metric_managers[n_hint], manual_metric_managers[n_hint])

            # post_metrics = [metric_manager.average_running_metric() for metric_manager in post_metric_managers]
            # manual_metrics = [metric_manager.average_running_metric() for metric_manager in manual_metric_managers]

        # # save metrics
        # for t in range(min(max_hint, len(post_metrics))):
        #     save_manager.write_log('(model ) Hint {} ::: {}'.format(t, post_metrics[t]))
        #     save_manager.write_log('(manual) Hint {} ::: {}'.format(t, manual_metrics[t]))
        # save_manager.save_metric(post_metrics, manual_metrics)

        if save_manager.config.save_test_prediction:
            save_manager.add_detect_pred_for_save()

    def forward_batch(self, batch, return_post_processing_pred=False):
        out, batch = self.model(batch)
        with torch.no_grad():
            # Post processing
            post_processing_pred = copy.deepcopy(out.pred)
            post_processing_pred.sargmax_coord = post_processing_pred.sargmax_coord.detach()
            post_processing_pred.heatmap = post_processing_pred.heatmap.detach()
            # for i in range(len(batch.hint.index)):
            #     if batch.hint.index[i] is not None:
            #         post_processing_pred.sargmax_coord[i, batch.hint.index[i]] = batch.label.coord[i, batch.hint.index[i]].detach().to(post_processing_pred.sargmax_coord.device)
            #         post_processing_pred.heatmap[i, batch.hint.index[i]] = batch.label.heatmap[i, batch.hint.index[i]].detach().to(post_processing_pred.heatmap.device)
        if return_post_processing_pred:
            return out, batch, post_processing_pred
        else:
            return out, batch


def main(save_manager):
    writer = None

    # Model initialization
    model = nn.DataParallel(get_model(save_manager), device_ids=[0])
    print(">> cuda is available: {}".format("true" if torch.cuda.is_available() else "false"))
    model.to(save_manager.config.MISC.device)

    # Calculate the number of model parameters
    n_params = 0
    for k, v in model.named_parameters():  # 遍歷model中的每一層，k是名稱，v是參數值
        n_params += v.reshape(-1).shape[0]  # v是一個tensor，reshape(-1)表示將v展平；shape[0]表示v展平後的元素個數。
    save_manager.write_log('Number of model parameters : {}'.format(n_params), 0)

    # metric_manager = MetricManager(save_manager)

    # Load parameters
    detector = Detector(model)
    detector.best_param, detector.best_epoch, detector.best_metric = save_manager.load_model()

    save_manager.write_log('Start Detecting...'.format(n_params), 4)
    # test_loader = get_dataloader(save_manager.config, 'test')
    input_image = cv2.imread(INPUT_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    input_image = cv2.resize(input_image, (256, 512))
    input_image = np.stack([input_image, input_image, input_image], axis=-1)
    input_image = np.transpose(input_image, (2, 0, 1))
    detector.detect(save_manager=save_manager, input_image=input_image, writer=writer)

    # del test_loader


if __name__ == '__main__':
    start_time = time.time()

    arg = parse_args()  # 解析傳入的參數
    save_manager = SaveManager(arg)  # 管理檔案的東西
    save_manager.write_log('Process Start ::: {}'.format(datetime.datetime.now(), n_mark=16))

    main(save_manager)

    end_time = time.time()
    save_manager.write_log('Process End ::: {} {:.2f} hours'.format(datetime.datetime.now(), (end_time - start_time) / 3600), n_mark=16)
    save_manager.write_log('Version ::: {}'.format(save_manager.config.version), n_mark=16)
