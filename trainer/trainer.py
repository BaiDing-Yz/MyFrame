import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import os

import data_loader as module_data
from util import *
import pandas as pd
import datetime


class Trainer(BaseTrainer):
    """
    训练器
    """
    def __init__(self,model,loss1,loss2,metrics,optimizer,data_loader,opt = None,
                 valid_data_loader=None,lr_scheduler=None,len_epoch=None):
        super().__init__(model,loss1,loss2,metrics,optimizer,opt)
        self.opt = opt
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.len_epoch = len(self.data_loader)
        if valid_data_loader is not None:
            self.len_val_epoch = len(self.valid_data_loader)
        

    def _eval_metrics(self, output, target):
#         acc_metrics = np.zeros(len(self.metrics))
#         for i, metric in enumerate(self.metrics):
#             acc_metrics[i] += metric(output, target)
        return self.metrics[0](output,target)
    
    def _train_epoch(self, epoch):
        self.model.train()
        
        total_loss = 0
        total_loss_day = 0
        total_loss_hour = 0
        
        total_loss_arr = []
        total_loss_day_arr = []
        total_loss_hour_arr = []
        
        total_metrics = np.zeros(3)
        trainloader_t = tqdm(self.data_loader,ncols=100)
        
        trainloader_t.set_description("Train Epoch: {}|{}  Batch size: {}  LR : {:.4}".format
                                      (epoch,self.opt.MAX_EPOCH,self.data_loader.batch_size,self.optimizer.param_groups[0]['lr']))
        for batch_idx, (inputs, targets_sign_day, targets_sign_hour, targets_ship_day, targets_ship_hour, targets_got_day, targets_got_hour, targets_dlved_day, targets_dlved_hour,payed_time,signed_time) in enumerate(trainloader_t): 
            batch_len = len(inputs)
            if self.use_gpu:
                inputs = inputs.cuda(self.opt.DEVICE_ID[0])
                targets_sign_day = targets_sign_day.cuda(self.opt.DEVICE_ID[0])
                targets_sign_hour = targets_sign_hour.cuda(self.opt.DEVICE_ID[0])
                targets_ship_day = targets_ship_day.cuda(self.opt.DEVICE_ID[0])
                targets_ship_hour = targets_ship_hour.cuda(self.opt.DEVICE_ID[0])
                targets_got_day = targets_got_day.cuda(self.opt.DEVICE_ID[0])
                targets_got_hour = targets_got_hour.cuda(self.opt.DEVICE_ID[0])
                targets_dlved_day = targets_dlved_day.cuda(self.opt.DEVICE_ID[0])
                targets_dlved_hour = targets_dlved_hour.cuda(self.opt.DEVICE_ID[0])
            self.optimizer.zero_grad()
            
            
            (output_FC_1_1, output_FC_2_1, output_FC_3_1, output_FC_4_1, output_FC_1_2,
             output_FC_2_2, output_FC_3_2, output_FC_4_2) = self.model(inputs)
#             (output_FC_1_1,output_FC_1_2) = self.model(inputs)
                
            output_FC_1_1 = output_FC_1_1.reshape(-1)
#             output_FC_2_1 = output_FC_2_1.reshape(-1)
#             output_FC_3_1 = output_FC_3_1.reshape(-1)
#             output_FC_4_1 = output_FC_4_1.reshape(-1)

            output_FC_1_2 = output_FC_1_2.reshape(-1)
#             output_FC_2_2 = output_FC_2_2.reshape(-1)
#             output_FC_3_2 = output_FC_3_2.reshape(-1)
#             output_FC_4_2 = output_FC_4_2.reshape(-1)

            loss_1_1 = self.loss2(output_FC_1_1, targets_sign_day.float())
#             loss_2_1 = self.loss1(output_FC_2_1, targets_ship_day.float())
#             loss_3_1 = self.loss1(output_FC_3_1, targets_got_day.float())
#             loss_4_1 = self.loss1(output_FC_4_1, targets_dlved_day.float())

            loss_1_2 = self.loss1(output_FC_1_2, targets_sign_hour.float())
#             loss_2_2 = self.loss1(output_FC_2_2, targets_ship_hour.float())
#             loss_3_2 = self.loss1(output_FC_3_2, targets_got_hour.float())
#             loss_4_2 = self.loss1(output_FC_4_2, targets_dlved_hour.float())
            
            
            loss_day = loss_1_1
            loss_hour = loss_1_2

#             loss_day = [loss_1_1,loss_2_1,loss_3_1,loss_4_1] * self.opt.LOSS_SCALE
#             loss_day  = 10 * np.sum(loss_day)
            
#             loss_hour = [loss_1_2,loss_2_2,loss_3_2,loss_4_2] * self.opt.LOSS_SCALE
#             loss_hour = np.sum(loss_hour)

            loss = 100 * loss_day + loss_hour
            loss.backward()
            
            total_loss_arr.append(loss.item())
            total_loss_day_arr.append(loss_day.item())
            total_loss_hour_arr.append(loss_hour.item())
            
            self.optimizer.step()
            
#             log = self._progress(batch_idx,loss.item())
#             print(log)

            total_loss += loss.item()
            total_loss_day += loss_day.item()
            total_loss_hour += loss_hour.item()
            
            
            pred_signed_time = []
            real_signed_time = []
            
#             print(loss_day.item())
#             print(loss_hour.item())
            for i in range(inputs.shape[0]):
                
                pred_time_day = output_FC_1_1[i]
                pred_time_hour = output_FC_1_2[i]
           
                temp_payed_time = payed_time[i]
                temp_payed_time = datetime.datetime.strptime(temp_payed_time, "%Y-%m-%d %H:%M:%S")
                temp_payed_time = temp_payed_time.replace(hour = int(pred_time_hour)%24)

                temp_pred_signed_time = temp_payed_time + relativedelta(days = int(pred_time_day))
                temp_pred_signed_time = temp_pred_signed_time.replace(hour = int(pred_time_hour)%24)
                temp_pred_signed_time = temp_pred_signed_time.replace(minute = 0)
                temp_pred_signed_time = temp_pred_signed_time.replace(second = 0)
                # temp_pred_signed_time.

                pred_signed_time.append(temp_pred_signed_time.strftime("%Y-%m-%d %H"))
                real_signed_time.append(signed_time[i])

#             (rankScore_result, onTimePercent_result, accuracy_result) = self._eval_metrics(real_signed_time, pred_signed_time)

            
            total_metrics += self._eval_metrics(real_signed_time, pred_signed_time)
            

        log = {
            'loss': total_loss / self.len_epoch,
#             'loss_day':total_loss_day / self.len_epoch,
#             'loss_hour':total_loss_hour / self.len_epoch,
            'rankScore': total_metrics[0]/ self.len_epoch,
            'onTimePercent':total_metrics[1]/ self.len_epoch,
            'accuracy':total_metrics[2]/ self.len_epoch,
        }
        if self.opt.DISPLAY and epoch < self.opt.DISPLAY_NUM:
            dirname = os.path.join(self.checkpoint_dirname,'epoch{}'.format(epoch))
            ensure_dir(dirname)
            plt.figure()
            plt.plot(total_loss_arr)
            plt.title('total_loss')
            plt.savefig(os.path.join(dirname, 'total_loss.png'))
            plt.figure()
            plt.plot(total_loss_day_arr)
            plt.title('total_loss_day')
            plt.savefig(os.path.join(dirname, 'total_loss_day.png'))
            plt.figure()
            plt.plot(total_loss_hour_arr)
            plt.title('total_loss_hour')
            plt.savefig(os.path.join(dirname, 'total_loss_hour.png'))
        

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        if self.opt.MINE:
            adjust_learning_rate(self.optimizer,epoch,self.opt)
        
        return log

            
            
    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_loss_day = 0
        total_val_loss_hour = 0
        total_val_metrics = np.zeros(3)
        valid_data_loader_t = tqdm(self.valid_data_loader,ncols=100)
        valid_data_loader_t.set_description("Val Epoch: {}|{}  Batch size: {}  ".format
                                      (epoch,self.opt.MAX_EPOCH,self.valid_data_loader.batch_size))
        with torch.no_grad():
            for batch_idx, (inputs, targets_sign_day, targets_sign_hour, targets_ship_day, targets_ship_hour, targets_got_day, targets_got_hour, targets_dlved_day, targets_dlved_hour,payed_time,signed_time) in enumerate(valid_data_loader_t):
                if self.use_gpu:
                    inputs = inputs.cuda(self.opt.DEVICE_ID[0])
                    targets_sign_day = targets_sign_day.cuda(self.opt.DEVICE_ID[0])
                    targets_sign_hour = targets_sign_hour.cuda(self.opt.DEVICE_ID[0])
                    targets_ship_day = targets_ship_day.cuda(self.opt.DEVICE_ID[0])
                    targets_ship_hour = targets_ship_hour.cuda(self.opt.DEVICE_ID[0])
                    targets_got_day = targets_got_day.cuda(self.opt.DEVICE_ID[0])
                    targets_got_hour = targets_got_hour.cuda(self.opt.DEVICE_ID[0])
                    targets_dlved_day = targets_dlved_day.cuda(self.opt.DEVICE_ID[0])
                    targets_dlved_hour = targets_dlved_hour.cuda(self.opt.DEVICE_ID[0])
                
                (output_FC_1_1, output_FC_2_1, output_FC_3_1, output_FC_4_1, output_FC_1_2,
             output_FC_2_2, output_FC_3_2, output_FC_4_2) = self.model(inputs.float())
#                 (output_FC_1_1,output_FC_1_2) = self.model(inputs)

                output_FC_1_1 = output_FC_1_1.reshape(-1)

                output_FC_1_2 = output_FC_1_2.reshape(-1)


                

                pred_signed_time = []
                real_signed_time = []

                for i in range(inputs.shape[0]):
                    pred_time_day = output_FC_1_1[i]
                    pred_time_hour = output_FC_1_2[i]
                
                    temp_payed_time = payed_time[i]
                    temp_payed_time = datetime.datetime.strptime(temp_payed_time, "%Y-%m-%d %H:%M:%S")
                    temp_payed_time = temp_payed_time.replace(hour =int(pred_time_hour)%24)

                    temp_pred_signed_time = temp_payed_time + relativedelta(days = int(pred_time_day))
                    temp_pred_signed_time = temp_pred_signed_time.replace(hour = int(pred_time_hour)%24)
                    temp_pred_signed_time = temp_pred_signed_time.replace(minute = 0)
                    temp_pred_signed_time = temp_pred_signed_time.replace(second = 0)
                    # temp_pred_signed_time.

                    pred_signed_time.append(temp_pred_signed_time.strftime("%Y-%m-%d %H"))
                    real_signed_time.append(signed_time[i])

    #             (rankScore_result, onTimePercent_result, accuracy_result) = self._eval_metrics(real_signed_time, pred_signed_time)


                total_val_metrics += self._eval_metrics(real_signed_time, pred_signed_time)

        log = {
            'val_rankScore': total_val_metrics[0]/ self.len_val_epoch,
            'val_onTimePercent':total_val_metrics[1]/ self.len_val_epoch,
            'val_accuracy':total_val_metrics[2]/ self.len_val_epoch,
        }
        return log

    
    def _progress(self, batch_idx,loss):
        current = batch_idx
        total = self.len_epoch
        log = '[{}/{} ({:.0f}%) Loss : {}]'.format(current, total, 100.0 * current / total,loss)
        return log
    
    
    def test(self):
        self.model.eval()
        testloader_t = tqdm(self.data_loader,ncols=100)
        testloader_t.set_description("Testing  Batch size: {}".format(self.data_loader.batch_size))
        pred_signed_time = []
        with torch.no_grad():
            for batch_idx, (inputs,payed_time) in enumerate(testloader_t): 
                if self.use_gpu:
                    inputs = inputs.cuda(self.opt.DEVICE_ID[0])
                
                (output_FC_1_1, output_FC_2_1, output_FC_3_1, output_FC_4_1, output_FC_1_2,
             output_FC_2_2, output_FC_3_2, output_FC_4_2) = self.model(inputs.float())

                output_FC_1_1 = output_FC_1_1.reshape(-1)

                output_FC_1_2 = output_FC_1_2.reshape(-1)
                
                

                for i in range(len(inputs)):
                    temp_payed_time = payed_time[i]
                    temp_payed_time = datetime.datetime.strptime(temp_payed_time, "%Y-%m-%d %H:%M:%S")
                    # temp_pred_signed_time = temp_payed_time + relativedelta(hours = pred_time_interval)

                    pred_time_day = output_FC_1_1[i]
                    pred_time_hour = output_FC_1_2[i]
                    temp_pred_signed_time = temp_payed_time + relativedelta(days = int(pred_time_day))
                    temp_pred_signed_time = temp_pred_signed_time.replace(hour = int(pred_time_hour)%24)    

                    # temp_pred_signed_time = temp_payed_time + relativedelta(hours = pred_time_interval)
                    pred_signed_time.append(temp_pred_signed_time.strftime('%Y-%m-%d %H'))

            # save predict result to txt file
            with open(self.opt.TEST_OUTPUT_PATH, 'w') as f:
                for res in pred_signed_time:
                    f.write(res + "\n")
                print(self.opt.TEST_OUTPUT_PATH,' has been created')
    
        


    
    
    
        
        
