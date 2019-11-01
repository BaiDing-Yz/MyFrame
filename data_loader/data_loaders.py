from base import BaseDataLoader
from torchvision import datasets, transforms
from data_loader.datasets import TrainSet,TestSet
from util import collate_function
from torch.utils.data.dataloader import default_collate

class TrainDataLoader(BaseDataLoader):
    def __init__(self,source_file,batch_size,opt = None,shuffle=False,validation_split=0.0,
                 num_workers=2,training=True,collate_fn = default_collate):
        self.source_file = source_file
        self.dataset = TrainSet(self.source_file, opt)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers,collate_fn = collate_fn)
        
        
        
class TestDataLoader(BaseDataLoader):
    def __init__(self,source_file,batch_size,opt = None,shuffle=False,validation_split=0.0,
                 num_workers=2,training=False,collate_fn = default_collate):
        self.source_file = source_file
        self.dataset = TestSet(self.source_file, opt)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers,collate_fn = collate_fn)
        
        

       
    
if __name__ == '__main__':
    config = Config()
    train_loader = module_data.TrainDataLoader(config.TRAIN_LIST, config.TRAIN_BATCH_SIZE,opt = config,validation_split = 0.1)