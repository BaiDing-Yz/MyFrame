from base import BaseConfig
import numpy as np

class Config(BaseConfig):
    def __init__(self):
        super(Config,self).__init__()
        # use GPU or not（GPU）
        self.USE_GPU = False
        self.DEVICE_ID = [1]
        self.Seed = 0


        #Model（模型）
        self.MODEL = 'Network'
        self.VERSION = '1'
        self.PUBLIC_LAYER = 1
        self.DAY_LAYER_NUM = 1
        self.HOUR_LAYER_NUM = 1

        #args（参数）
        self.FINETUNE = False
        self.NUMCLASSES = 16516


        #Metric(评价标准）
        self.METRIC = ['calculateAllMetrics']

        #Loss
        self.LOSS1 = 'MSELoss'
        self.LOSS2 = 'My_MSE_loss'
        self.GAMMA1 = 2
        self.GAMMA2 = 2
        self.LOSS3 = 'MyLoss'

        self.HOUR_SCALE = 1
        self.MIDDLE_SCALE = 5
        self.MIDDLE_TWO_SCALE = 10
        self.DAY_SCALE = 20
        self.SCALES = [self.HOUR_SCALE,self.MIDDLE_SCALE,self.MIDDLE_TWO_SCALE,self.DAY_SCALE]
        self.SCALE_NUM = 2

        self.LOSS_SCALE = np.array([1, 0.15, 0.15, 0.15])
        
        self.DH_SCALE = [1,50]
        

        #Visualization（可视化）
        self.DISPLAY = False
        self.DISPLAY_NUM = np.inf


        #Dataset
        self.DATASET_NORMALIZATION = False

        #clip_data
        self.uid_range            =   1505257
        self.plat_form_range      =   4
        self.biz_type_range       =   6
        self.product_id_range     =   51805   
        self.cate1_id_range       =   25
        self.cate2_id_range       =   244
        self.cate3_id_range       =   1429
        self.seller_uid_range     =   1000 
        self.company_name_range   =   929
        self.rvcr_prov_name_range =   31
        self.rvcr_city_name_range =   370
        self.day_range = 31
        self.hour_range = 24
        
        self.EMBEDDING_NUM = 14




        #DataLoader（数据加载）
        self.TRAIN_LIST = '../data/SeedCup2019_pre/SeedCup_pre_train.csv'
        self.TEST_LIST = '../data/SeedCup2019_pre/SeedCup_pre_test.csv'

        self.NUM_WORKERS = 4  

        self.TRAIN_BATCH_SIZE = 128  # batch size
        self.TEST_BATCH_SIZE = 128  # batch size


        #Checkpoints（模型、日志保存）
        self.CHECKPOINTS_ROOT = 'checkpoints'



        #Optimizer（优化器）
        self.OPTIMIZER = 'Adam'
        self.LR = 0.01  # initial learning rate
        self.WEIGHT_DECAY = 5e-4

        #LR Scheduler（学习率下降）
        self.MINE = False
        self.LR_INCREASE = 1.3
        self.INCREASE_BOTTOM = 10
        self.LR_SCHEDULER = 'StepLR'
        self.LR_STEP = 100
        self.LR_DECAY = 0.1    

        #Trainer（训练器）
        self.MAX_EPOCH = 300
        self.PRINT_FREQ = np.inf  # print info every N batch
        self.SAVE_PERIOD = 10
        self.EARLY_STOP = 300
        self.START_EPOCH = 0

        self.BEST_MNT = True
        self.MONITOR_CONDITION="val_onTimePercent"
        self.MONITOR_CONDITION_VALUE = 0.98
        self.MONITOR = "min val_rankScore"

        #Resume Checkpoint（模型加载）
        self.TEST = False
        self.RESUME = False
        self.RESUME_MODEL = 'Network'
        self.RESUME_FORMAT = 'checkpoint-epoch{}_V{}.pth'
        self.RESUME_VERSION = '4'
        self.RESUME_EPOCH = 100
        self.RESUME_DIR = self.RESUME_FORMAT.format(self.RESUME_EPOCH,self.RESUME_VERSION)



        #Trainer
        self.EMBEDDING_DIM      =       100
        self.LINER_HID_SIZE     =       1024
        self.INPUT_SIZE         =       11
        self.OUTPUT_DIM              =   1
        self.OUTPUT_TIME_INTERVAL_2  =   1
        self.OUTPUT_TIME_INTERVAL_3  =   1
        self.OUTPUT_TIME_INTERVAL_4  =   1
        self.LOSS_1_WEIGHT           =   1

        
        time = 13
        #提交文件
        self.TEST_OUTPUT_PATH = '../data/SeedCup2019_pre/test' + '_V'+self.RESUME_VERSION + '_' + str(time) + '.txt'