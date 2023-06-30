import time
import yaml

import numpy as np
import random
import torch
import torch.nn as nn
import copy
from munch import Munch
from misc.metric import MetricManager
from tqdm.auto import tqdm

from util import SaveManager
from model.model import IKEM
from dataset import get_dataloader
from misc.metric import MetricManager
from misc.optimizer import get_optimizer
from misc.train import Trainer

CONFIG_PATH = ".\config\config.yaml"
CHECKPOINT_PATH = "..\save\model.pth"


class Tester(object):
    def __init__(self, model, metric_manager):
        self.model = model

        self.metric_manager = metric_manager
        self.best_epoch = None
        self.best_param = None
        self.best_metric = self.metric_manager.init_best_metric()

        self.patience = 0

    def test(self, save_manager, test_loader, writer=None):
        # To reduce GPU usage, load params on cpu and upload model on gpu(params on gpu also need additional GPU memory)
        self.model.cpu()
        self.model.load_state_dict(self.best_param)
        self.model.to(save_manager.config.MISC.device)


        save_manager.write_log('Best model at epoch {} loaded'.format(self.best_epoch))
        self.model.eval()
        with torch.no_grad():
            post_metric_managers = [MetricManager(save_manager) for _ in range(save_manager.config.Dataset.num_keypoint+1)]
            manual_metric_managers = [MetricManager(save_manager) for _ in range(save_manager.config.Dataset.num_keypoint+1)]

            for i, batch in enumerate(tqdm(test_loader)):
                batch = Munch.fromDict(batch)
                batch.detecting = False
                batch.is_training = False
                max_hint = 10+1
                for n_hint in range(max_hint):
                    # 0~max hint

                    ## model forward
                    out, batch, post_processing_pred = self.forward_batch(batch, metric_flag=True, average_flag=False, metric_manager=post_metric_managers[n_hint], return_post_processing_pred=True)
                    if save_manager.config.Model.use_prev_heatmap:
                        batch.prev_heatmap = out.pred.heatmap.detach()

                    ## manual forward
                    if n_hint == 0:
                        manual_pred = copy.deepcopy(out.pred)
                        manual_pred.sargmax_coord = manual_pred.sargmax_coord
                        manual_pred.heatmap = manual_pred.heatmap
                        manual_hint_index = []
                    else:
                        # manual prediction update
                        for k in range(manual_hint_index.shape[0]):
                            manual_pred.sargmax_coord[k, manual_hint_index[k]] = batch.label.coord[k, manual_hint_index[k]].to(manual_pred.sargmax_coord.device)
                            manual_pred.heatmap[k, manual_hint_index[k]] = batch.label.heatmap[k, manual_hint_index[k]].to(manual_pred.heatmap.device)
                    manual_metric_managers[n_hint].measure_metric(manual_pred, batch.label, batch.pspace, metric_flag=True, average_flag=False)

                    # ============================= model hint =================================
                    worst_index = self.find_worst_pred_index(batch.hint.index, post_metric_managers, save_manager, n_hint)
                    # hint index update
                    if n_hint == 0:
                        batch.hint.index = worst_index # (batch, 1)
                    else:
                        batch.hint.index = torch.cat((batch.hint.index, worst_index.to(batch.hint.index.device)), dim=1) # ... (batch, max hint)

                    #
                    if save_manager.config.Model.use_prev_heatmap_only_for_hint_index:
                        new_prev_heatmap = torch.zeros_like(out.pred.heatmap)
                        for j in range(len(batch.hint.index)):
                            new_prev_heatmap[j, batch.hint.index[j]] = out.pred.heatmap[j, batch.hint.index[j]]
                        batch.prev_heatmap = new_prev_heatmap
                    # ================================= manual hint =========================
                    worst_index = self.find_worst_pred_index(manual_hint_index, manual_metric_managers, save_manager, n_hint)
                    # hint index update
                    if n_hint == 0:
                        manual_hint_index = worst_index  # (batch, 1)
                    else:
                        manual_hint_index = torch.cat((manual_hint_index, worst_index.to(manual_hint_index.device)), dim=1)  # ... (batch, max hint)

                    # save result
                    if save_manager.config.save_test_prediction:
                        save_manager.add_test_prediction_for_save(batch, post_processing_pred, manual_pred, n_hint, post_metric_managers[n_hint], manual_metric_managers[n_hint])

            post_metrics = [metric_manager.average_running_metric() for metric_manager in post_metric_managers]
            manual_metrics = [metric_manager.average_running_metric() for metric_manager in manual_metric_managers]

        # save metrics
        for t in range(min(max_hint, len(post_metrics))):
            save_manager.write_log('(model ) Hint {} ::: {}'.format(t, post_metrics[t]))
            save_manager.write_log('(manual) Hint {} ::: {}'.format(t, manual_metrics[t]))
        save_manager.save_metric(post_metrics, manual_metrics)

        if save_manager.config.save_test_prediction:
            save_manager.save_test_prediction()

    def forward_batch(self, batch, metric_flag=False, average_flag=True, metric_manager=None, return_post_processing_pred=False):
        out, batch = self.model(batch)
        with torch.no_grad():
            if metric_manager is None:
                self.metric_manager.measure_metric(out.pred, batch.label, batch.pspace, metric_flag, average_flag)
            else:
                # post processing
                post_processing_pred = copy.deepcopy(out.pred)
                post_processing_pred.sargmax_coord = post_processing_pred.sargmax_coord.detach()
                post_processing_pred.heatmap = post_processing_pred.heatmap.detach()
                for i in range(len(batch.hint.index)): # for 문이 batch에 대해서 도는중 i번째 item
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


def test(config):
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # Model
    model = IKEM(config).to(device)
    model.eval()

    # Calculate the number of model parameters
    n_params = 0
    for k, v in model.named_parameters():  # 遍歷model的每一層，k是名稱，v是參數值
        n_params += v.reshape(-1).shape[0]  # v是一個tensor，reshape(-1)表示將v展平；shape[0]表示v展平後的元素個數。
    print('Number of model parameters : {}'.format(n_params))

    metric_manager = MetricManager(save_manager)
    tester = Tester(model, metric_manager)

    # Load model parameters
    save = torch.load(CHECKPOINT_PATH)
    tester.best_param = save['model']
    tester.best_epoch = save['epoch']
    tester.best_metric = None

    print('Start Test Evaluation...'.format(n_params))
    test_loader = get_dataloader(save_manager.config, 'test')
    tester.test(save_manager=save_manager, test_loader=test_loader, writer=writer)

    del test_loader



if __name__ == '__main__':
    start_time = time.time()
    print("\n>> Start Program --- {} \n".format(start_time))

    # Read config
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)  # 讀入yaml檔
    config = Munch.fromDict(config)  # dictionary轉munch

    # Testing
    test(config)

    end_time = time.time()
    print("\n>> End Program --- {} \n".format(end_time))
