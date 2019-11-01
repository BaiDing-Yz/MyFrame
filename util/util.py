import json
from pathlib import Path
from datetime import datetime
from itertools import repeat
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
from config import *
import os
import random
from torchsummary import summary


def ensure_dir(dirname):
    '''
    确定存在dirname文件夹，若不存在创建该文件夹
    :param dirname: 文件夹的绝对路径
    :return:
    '''
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    '''
    读取json文件
    :param fname:json文件路径
    :return:
    '''
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    '''
    将content的内容写到fanme文件中
    :param content: 将要写入json文件的内容
    :param fname: json文件的绝对路径
    :return:
    '''
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


class Timer:
    '''
    用于测量时间
    '''
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()


def collate_function(batch):
    '''
    更改batch的内容
    :param batch:
    :return:
    '''
    batch = [(inputs, targets_sign_day, targets_sign_hour, targets_ship_day, targets_ship_hour, targets_got_day, targets_got_hour, targets_dlved_day, targets_dlved_hour,payed_time,signed_time) 
             for (inputs, targets_sign_day, targets_sign_hour, targets_ship_day, targets_ship_hour, targets_got_day, targets_got_hour, targets_dlved_day, targets_dlved_hour,payed_time,signed_time) in batch if (targets_sign_day is not np.NaN) and (targets_sign_hour is not np.NaN) and(targets_ship_day is not np.NaN) and(targets_ship_hour is not np.NaN) and(targets_got_day is not np.NaN) and(targets_got_hour is not np.NaN) and(targets_dlved_day is not np.NaN) and(targets_dlved_hour is not np.NaN)]
#     return torch.tensor(batch)
    return batch




def adjust_learning_rate(optimizer, epoch,config):
    '''
    自定义自适应学习率变化，退火
    :param optimizer: 优化器
    :param epoch: 学习率增加与降低的节点
    :param config: 包含LR_INCREASE和LR_DECAY
    :return:
    '''
     
    nowLR = optimizer.param_groups[0]['lr']
    LR = nowLR
    if epoch <= config.INCREASE_BOTTOM:
        LR = LR * config.LR_INCREASE
    else:
        if (epoch-config.INCREASE_BOTTOM) % config.LR_STEP == 0:
            LR = LR * config.LR_DECAY
 
    for param_group in optimizer.param_groups:
        param_group['lr'] = LR
        
        
def show_config(checkpoint_dirname,filename):
    '''
    展示config文件的内容
    :param checkpoint_dirname: 加载checkpint_dirname文件夹下某文件的config文件
    :param filename: 文件名
    :return:
    '''
    filename = os.path.join(checkpoint_dirname,filename)
    message = "There's not checkpoint"
    assert os.path.exists(filename),message
    checkpoint = torch.load(filename)
    config = checkpoint['config']
    return config


def setup_seed(seed):
    '''
    设置随机数
    :param seed:随机数
    :return:
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def model_visual(your_model,input_shape):
    '''
    用于展示your_model的所有输入输出shape
    :param your_model: 用于展示的Model
    :param inputs_shape: Model的输入
    :return:
    '''
    print(summary(your_model, input_size=input_shape))
