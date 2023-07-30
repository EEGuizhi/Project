import os
import torch
import random
import numpy as np
import pandas as pd


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


def save_model(path:str, epoch:int, model:torch.nn.Module, optimizer:torch.optim.Optimizer, train_loss:list=None, val_loss:list=None):
    print("Saving model..")
    save_dict = {
        "model": model.state_dict(),
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss
    }
    torch.save(save_dict, path)


def choose_hint_index(pred_coords:torch.Tensor, label_coords:torch.Tensor):
    index = find_worst_index(pred_coords, label_coords, size=10)
    index = np.random.choice(np.array(index), size=None)
    return index


def find_worst_index(pred_coords:torch.Tensor, label_coords:torch.Tensor, size:int=1):
    # Dim of inputs = (68, 2)
    diff_coords = torch.pow(pred_coords - label_coords, 2)
    diff_coords = torch.sum(diff_coords, dim=-1)
    return torch.topk(diff_coords, size).indices.tolist()


def get_batch_MRE(pred_coords:torch.Tensor, label_coords:torch.Tensor):
    batch_mre = 0
    for s in range(pred_coords.shape[0]):
        batch_mre += get_MRE(pred_coords[s], label_coords[s])
    batch_mre /= pred_coords.shape[0]
    return batch_mre


def get_MRE(pred_coords:torch.Tensor, label_coords:torch.Tensor):
    # Dim of inputs = (68, 2)
    num_of_keypoints = pred_coords.shape[0]
    diff_coords = torch.pow(pred_coords - label_coords, 2)
    diff_coords = torch.sum(diff_coords, dim=-1)
    diff_coords = torch.pow(diff_coords, 0.5)
    mre = torch.sum(diff_coords).cpu() / num_of_keypoints
    return mre.item()


def is_worth_to_save(train_loss:tuple, val_loss:tuple, saved_train_loss:list, saved_val_loss:list):
    """
    Parameters:
    ---
        - `train_loss: (pred1_loss, pred2_loss)`
        - `val_loss: (pred1_loss, pred2_loss)`
        - `saved_train_loss: [lowest_train_loss(tuple), largest_gap_train_loss(tuple)]`
        - `saved_val_loss: [lowest_val_loss(tuple), largest_gap_val_loss(tuple)]`

    Returns:
    ---
        - `better pred acc: (bool)`
        - `larger gap between pred1 & pred2: (bool)`
        - `saved_train_loss: (list)`
        - `saved_val_loss: (list)`
    """
    # No Saved model
    if saved_train_loss is None:
        return True, True, [train_loss, train_loss], [val_loss, val_loss]

    # Init
    better_pred = False
    larger_gap = False

    # Better pred(=pred1) acc
    if train_loss[0] < saved_train_loss[0][0] and val_loss[0] < saved_val_loss[0][0] and val_loss[0] < val_loss[1]:
        better_pred = True
        saved_train_loss[0], saved_val_loss[0] = train_loss, val_loss

    # Larger gap between pred1 & pred2  and  pred2 must better than pred1
    if (val_loss[0] - val_loss[1]) < (saved_val_loss[1][0] - saved_val_loss[1][1]) and val_loss[1] < saved_val_loss[0][0]:
        larger_gap = True
        saved_train_loss[1], saved_val_loss[1] = train_loss, val_loss

    return better_pred, larger_gap, saved_train_loss, saved_val_loss


def write_log(
        file_path:str, dataframe:pd.DataFrame, epoch:int, train_P1Loss:float, train_P2Loss:float,
        val_P1Loss:float, val_P2Loss:float, val_P1MRE:float, val_P2MRE:float
    ):

    # Init
    if dataframe is None:  # 重複讀寫很花時間
        if os.path.exists(file_path):
            dataframe = pd.read_csv(file_path)
        else:
            dataframe = pd.DataFrame({
                "Epoch": [],
                "Train 1stPred. Loss": [],
                "Train 2ndPred. Loss": [],
                "Val 1stPred. Loss": [],
                "Val 2ndPred. Loss": [],
                "Val 1stPred. MRE": [],
                "Val 2ndPred. MRE": []
            })

    dataframe = pd.concat([
        dataframe,
        pd.DataFrame({
            "Epoch": [epoch],
            "Train 1stPred. Loss": [train_P1Loss],
            "Train 2ndPred. Loss": [train_P2Loss],
            "Val 1stPred. Loss": [val_P1Loss],
            "Val 2ndPred. Loss": [val_P2Loss],
            "Val 1stPred. MRE": [val_P1MRE],
            "Val 2ndPred. MRE": [val_P2MRE]
        })
    ], ignore_index=True)

    dataframe.to_csv(file_path, index=False)
    return dataframe
