from pytz import timezone
import time
import argparse
import datetime

import numpy as np
import random
import torch
import torch.nn as nn

# 下方是作者自己寫的
from util import SaveManager, TensorBoardManager
from model import get_model
from dataset import get_dataloader
from misc.metric import MetricManager
from misc.optimizer import get_optimizer
from misc.train import Trainer


torch.set_num_threads(4)  # 設置在CPU上平行運算時所佔用的線程數

def set_seed(config):
    torch.backends.cudnn.deterministic = True  # 卷積的算法會是固定算法 (?
    torch.backends.cudnn.benchmark = False  # 設為True的話"可能"可以加速:https://zhuanlan.zhihu.com/p/73711222
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    return

def parse_args():
    '''
    解析傳入的參數。
    詳細了解:https://shengyu7697.github.io/python-argparse/
    '''
    parser = argparse.ArgumentParser(description='TMI experiments')  # 創建
    parser.add_argument('--config', type=str, help='config name, required')  # 接在"--config"後面的字串參數會存到parser.config
    parser.add_argument('--seed', type=int, default=42, help='random seed')  # 將亂數種子固定在42確保每次運作結果相同，設為42的原因：https://www.google.com.tw/search?q=the+answer+to+life+the+universe+and+everything&oq=the+answer+to+life+the+universe+and+everything
    parser.add_argument('--only_test_version', type=str, default=None, help='If activated, there is no training. The number is the experiment number. => load & test model')
    parser.add_argument('--save_test_prediction', action='store_true', default=False, help='If activated, save test predictions at save path')
    arg = parser.parse_args()  # 開始解析
    set_seed(arg)
    return arg



def main(save_manager):
    writer = None

    # model initialization
    device_ids = list(range(len(save_manager.config.MISC.gpu.split(','))))
    model = nn.DataParallel(get_model(save_manager), device_ids=device_ids)
    print(">> cuda is available: {}".format("true" if torch.cuda.is_available() else "false"))
    model.to(save_manager.config.MISC.device)

    # calculate the number of model parameters
    n_params = 0
    for k, v in model.named_parameters():  # 遍歷model中的每一層，k是名稱，v是參數值
        n_params += v.reshape(-1).shape[0]  # v是一個tensor，reshape(-1)表示將v展平；shape[0]表示v展平後的元素個數。
    save_manager.write_log('Number of model parameters : {}'.format(n_params), 0)

    # optimizer initialization
    optimizer = get_optimizer(save_manager.config.Optimizer, model)

    metric_manager = MetricManager(save_manager)
    trainer = Trainer(model, metric_manager)

    if not save_manager.config.only_test_version:
        # dataloader
        train_loader = get_dataloader(save_manager.config, 'train')
        val_loader = get_dataloader(save_manager.config, 'val')

        # training
        save_manager.write_log('Start Training...'.format(n_params), 4)
        trainer.train(
            save_manager=save_manager,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            writer=writer
        )
        # deallocate data loaders from the memory
        del train_loader
        del val_loader

    trainer.best_param, trainer.best_epoch, trainer.best_metric = save_manager.load_model()

    save_manager.write_log('Start Test Evaluation...'.format(n_params), 4)
    test_loader = get_dataloader(save_manager.config, 'test')
    trainer.test(save_manager=save_manager, test_loader=test_loader, writer=writer)
    del test_loader


if __name__ == '__main__':
    start_time = time.time()
    print("\n>> Start Program --- {} \n".format(start_time))

    arg = parse_args()  # 解析傳入的參數
    save_manager = SaveManager(arg)  # 管理檔案的東西
    main(save_manager)

    end_time = time.time()
    print("\n>> End Program --- {} \n".format(end_time))
