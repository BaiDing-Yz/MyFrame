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



if __name__ == '__main__':
    config = Config()
    config.TEST = True
    test_loader = module_data.TestDataLoader(config.TEST_LIST, config.TEST_BATCH_SIZE,opt = config)
 
    model = getattr(module_model,config.RESUME_MODEL)(config)
    
    trainer = Trainer(model, None,None, None, None,
                      opt=config,
                      data_loader=test_loader,
                      valid_data_loader=None,
                      lr_scheduler=None)
    
    print('start test')
    trainer.test()