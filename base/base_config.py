import numpy as np

class BaseConfig(object):
    def __init__(self):
        # 整体参数
        self.ENV = 'default'

        # use GPU or not（GPU）
        self.USE_GPU = True  
        self.DEVICE_ID = [1]
        self.Seed = 0

        #Model（模型）
        self.MODEL = 'resnet_face18'
        self.VERSION = '0'
        #args（参数）
        self.FINETUNE = False

        #Metric(评价标准）
        self.METRIC = ['accuracy']

        #Loss
        self.LOSS = 'focal_loss'

        #Visualization（可视化）
        self.DISPLAY = False
        self.DISPLAY_NUM = np.inf

        #DataLoader（数据加载）
        self.TRAIN_ROOT = '../data'
        self.TEST_ROOT = '../data'
        self.NUM_WORKERS = 4  

        self.TRAIN_BATCH_SIZE = 256  # batch size
        self.TEST_BATCH_SIZE = 60

        #Checkpoints（模型、日志保存）
        self.CHECKPOINTS_ROOT = 'checkpoints'

        #Optimizer（优化器）
        self.OPTIMIZER = 'sgd'
        self.LR = 0.05  # initial learning rate
        self.WEIGHT_DECAY = 5e-4

        #LR Scheduler（学习率下降）
        #### 自定义学习率下降方法
        self.MINE = False
        self.LR_INCREASE = 1.3
        self.INCREASE_BOTTOM = 10
        #### torch 自带学习率下降方法
        self.LR_SCHEDULER = 'StepLR'
        self.LR_STEP = 5
        self.LR_DECAY = 0.5


        #Trainer（训练器）
        self.MAX_EPOCH = 50
        self.PRINT_FREQ = np.inf  # print info every N batch
        self.SAVE_PERIOD = 10
        self.EARLY_STOP = np.inf
        self.START_EPOCH = 0

        self.BEST_MNT = True
        self.MONITOR_CONDITION = "val_onTimePercent"
        self.MONITOR_CONDITION_VALUE = 0.98
        self.MONITOR = "min val_rankScore"  # or "off"

        #Resume Checkpoint（模型加载）
        self.TEST = False     #用于测试模型的加载
        self.RESUME = False   #用于训练时模型的加载
        self.RESUME_MODEL = 'Network'
        self.RESUME_FORMAT = 'checkpoint-epoch{}_V{}.pth'
        self.RESUME_VERSION = '0'
        self.RESUME_EPOCH = 100
        self.RESUME_DIR = self.RESUME_FORMAT.format(self.RESUME_EPOCH, self.RESUME_VERSION)

        #LOG
        self.LOG_NAME = 'output.log'
        self.time = 0
        # 提交文件
        self.TEST_OUTPUT_PATH = '../data/SeedCup2019_pre/test' + '_V' + self.RESUME_VERSION + '_' + str(self.time) + '.txt'

    def list_all_member(self):
        '''
        print所有的参数
        '''
        for name,value in vars(self).items():
            print('%s=%s'%(name,value))
            
    def write_to_txt(self,file):
        """
        将参数输出到file文件中
        """
        for name,value in vars(self).items():
            string = '{}={}\n'.format(name,value)
            file.write(string)
        string = '\n\n\n'
        file.write(string)

