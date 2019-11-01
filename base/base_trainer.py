import torch
import torch.nn as nn
from abc import abstractmethod
from numpy import inf
from util import *
import os
from .base_config import BaseConfig


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, loss1,loss2, metrics, optimizer, config):
        self.config = config
        self.use_gpu = False
        self.model = model
        if config.USE_GPU:
            if not torch.cuda.is_available():
                print("There's no GPU is available , Now Automatically converted to CPU device")
            else:
                message = "There's no GPU is available"
                assert len(config.DEVICE_ID) > 0,message
                device_ids = config.DEVICE_ID
                self.model = self.model.cuda(device_ids[0])
                if len(device_ids) > 1:
                    self.model = nn.DataParallel(model, device_ids=device_ids)
                self.use_gpu = True
        self.loss1 = loss1
        self.loss2 = loss2
        self.metrics = metrics
        self.optimizer = optimizer

        self.epochs = config.MAX_EPOCH
        self.checkpoint_dir = config.CHECKPOINTS_ROOT
        
        self.save_period = config.SAVE_PERIOD
        self.monitor = config.MONITOR
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = config.EARLY_STOP
        self.start_epoch = self.config.START_EPOCH
        
        self.checkpoint_dirname = os.path.join(self.checkpoint_dir,self.config.MODEL+'_v'+self.config.VERSION)
        self.test_dirname = os.path.join(self.checkpoint_dir,self.config.RESUME_MODEL+'_v'+self.config.RESUME_VERSION)
        ensure_dir(self.checkpoint_dirname)
        
        if self.config.RESUME:
            self.load_model()
        elif self.config.TEST:
            self.load_test_checkpoint()
        else:
            self.model.initialize_weight()
            

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        f = self.file_open(self.config.LOG_NAME)
        self.config.write_to_txt(f)
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    log.update({mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({'val_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                else:
                    log[key] = value
            self.file_write(f,log)
            # print logged informations to the screen
            for key, value in log.items():
                print('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = self.config.BEST_MNT
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved and (log[self.config.MONITOR_CONDITION] > self.config.MONITOR_CONDITION_VALUE):
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                    self._save_checkpoint(epoch, save_best=best)
                else:
                    best = False
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break
            
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
    
    
    def test(self):
        pass

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dirname,self.opt.RESUME_FORMAT.format(epoch,self.opt.VERSION))
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dirname , 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path,best = False):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        if best:
            filename = os.path.join(self.test_dirname,'model_best.pth')
        else:
            filename = os.path.join(self.test_dirname,resume_path)
            
        message = "There's not checkpoint"
        assert os.path.exists(filename),message
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(filename)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config'].MODEL != self.config.MODEL:
            print("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        
        else:
            self.model.load_state_dict(checkpoint['state_dict'])
        if checkpoint['config'].OPTIMIZER != self.config.OPTIMIZER:
            print("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
        
        
    def load_model(self,best = False):
        self._resume_checkpoint(self.config.RESUME_DIR,best)
        
        
    def _test_checkpoint(self,resume_path,best = False):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        if best:
            filename = os.path.join(self.test_dirname,'model_best.pth')
        else:
            filename = os.path.join(self.test_dirname,resume_path)
        message = "There's not checkpoint"
        assert os.path.exists(filename),message
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(filename)
        # load architecture params from checkpoint.
        if checkpoint['config'].MODEL != self.config.MODEL:
            print("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        else:
            self.model.load_state_dict(checkpoint['state_dict'])
            
            
    def load_test_checkpoint(self,best=False):
        self._test_checkpoint(self.config.RESUME_DIR,best)
            
            
            
    def file_open(self,file):
        filename = os.path.join(self.checkpoint_dirname,file)
        f = open(filename,'w')
        return f
    
    def file_write(self,file,log):
        string = ''
        for key, value in log.items():
            string += key + '\t:'+str(value) +'\n'
        string += '\n\n\n'
        file.write(string)
        
        
    
        
