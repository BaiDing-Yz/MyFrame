import os
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import random
import time
from config import Config
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from test import *
from tqdm import tqdm

import data_loader as module_data
import loss as module_loss
import metric as module_metric
import model as module_model
import torch.nn as nn_loss

from trainer import Trainer

from util import *

if __name__ == '__main__':
    config = Config()
    setup_seed(config.Seed)
    config.TEST = False
    train_loader = module_data.TrainDataLoader(config.TRAIN_LIST, config.TRAIN_BATCH_SIZE,opt = config,validation_split = 0.1)
    metrics = [getattr(module_metric, met) for met in config.METRIC]
    valid_loader = train_loader.split_validation()
    
    model = getattr(module_model,config.MODEL)(config)
    loss1 = getattr(nn_loss,config.LOSS1)()
    loss2 = getattr(module_loss,config.LOSS2)(scale = config.SCALES[config.SCALE_NUM])
#     loss3 = getattr(module_loss,config.LOSS3)(config.SCALES[config.SCALE_NUM],config.DH_SCALE)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = getattr(torch.optim,config.OPTIMIZER)(model.parameters(),
                                                      lr = config.LR, weight_decay=config.WEIGHT_DECAY)
    
    if not config.MINE:
        lr_scheduler = getattr(torch.optim.lr_scheduler,config.LR_SCHEDULER)(optimizer,step_size = config.LR_STEP,gamma = config.LR_DECAY)
    else:
        lr_scheduler = None
    trainer = Trainer(model, loss1,loss2, metrics, optimizer,
                      opt=config,
                      data_loader=train_loader,
                      valid_data_loader=valid_loader,
                      lr_scheduler=lr_scheduler)
    
    print('start train')
    trainer.train()