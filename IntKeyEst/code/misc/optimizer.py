import numpy as np
from torch import optim

def get_optimizer(config, model):
    optimizer = base_optimizer(config, model)
    return optimizer

class base_optimizer(object):
    def __init__(self, config, model):
        super(base_optimizer, self).__init__()

        # Optimizer
        if config.optimizer == 'Adam':  # https://medium.com/%E9%9B%9E%E9%9B%9E%E8%88%87%E5%85%94%E5%85%94%E7%9A%84%E5%B7%A5%E7%A8%8B%E4%B8%96%E7%95%8C/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92ml-note-sgd-momentum-adagrad-adam-optimizer-f20568c968db
            self.optimizer = optim.Adam(model.parameters(), lr=config.lr)
        elif config.optimizer == 'Adadelta':
            self.optimizer = optim.Adadelta(model.parameters(), lr=config.lr)
        else:
            raise

        # LR Scheduler https://machinelearningmastery.com/using-learning-rate-schedule-in-pytorch-training/
        if config.scheduler == 'ReduceLROnPlateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', verbose=True)
        elif config.scheduler == 'StepLR':
            from torch.optim.lr_scheduler import StepLR
            self.scheduler = StepLR(self.optimizer, 100, gamma=0.1, last_epoch=-1)
        else:
            self.scheduler = None

    def update_model(self, loss):
        self.optimizer.zero_grad()

        if np.isnan(loss.item()):
            print('\n\n\nERROR::: THE LOSS IS NAN\n\n\n')
            raise()
        else:
            loss.backward()
            self.optimizer.step()
        return None

    def scheduler_step(self, metric):
        if self.scheduler is not None:
            self.scheduler.step(metric)
        return None
