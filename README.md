## 代码结构

```
Frame
│   train.py （训练程序）
│   test.py  （测试程序）    
|   model_visual  (模型测试文件)
└───base
│   │   __init__.py
│   │   base_config.py         #配置文件的父类
│   │   base_data_loader.py    #数据加载的父类
│   │   base_model.py          #模型文件的父类
│   │   base_trainer.py        #训练文件的父类
└───config （配置文件）
│   │   __init__.py
│   │   config.py              #配置文件
└───checkpoints （模型保存的位置）
|   |   ....
└───data_loader （数据加载）
│   │   __init__.py
│   │   data_loaders.py        #数据加载器
│   │   datasets.py            #数据处理
└───loss    (自定义loss)
│   │   __init__.py
│   │   loss.py                #自定义loss
└───metric （评价标准）
│   │   __init__.py
│   │   metric.py              #模型评价标准
└───model   (存储各种模型)
│   │   __init__.py
│   └───Baseline               #模型文件
│       │   __init__.py
│       │   baseline.py 
└───traner   (训练器)
│   │   __init__.py
│   │   trainer.py (包含train和test程序)  #修改其中的train_epoch 和 test_epoch 进行训练
└───util   (各种工具函数)
│   │   __init__.py
│   │   util.py


```



## 模型训练

修改config文件的相应信息后，运行

```
python train.py
```

## 模型测试

修改config文件的相应信息后，运行

```
python test.py
```





**本框架文件将会不定期进行更新**