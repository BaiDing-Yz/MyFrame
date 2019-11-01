import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.functional as F
import sys
import numpy as np

'''
design loss function
'''

            
class Network(BaseModel):
    def __init__(self, opt):
        super(Network, self).__init__()

        # self.encoder_uid       = nn.Embedding(opt.uid_range, opt.EMBEDDING_DIM)
        self.encoder_plat_form = nn.Embedding(opt.plat_form_range, opt.EMBEDDING_DIM)
        self.encoder_biz_type = nn.Embedding(opt.biz_type_range, opt.EMBEDDING_DIM)
        self.encoder_product_id = nn.Embedding(opt.product_id_range, opt.EMBEDDING_DIM)

        self.encoder_cate1_id = nn.Embedding(opt.cate1_id_range, opt.EMBEDDING_DIM)
        self.encoder_cate2_id = nn.Embedding(opt.cate2_id_range, opt.EMBEDDING_DIM)
        self.encoder_cate3_id = nn.Embedding(opt.cate3_id_range, opt.EMBEDDING_DIM)

        self.encoder_seller_uid = nn.Embedding(opt.seller_uid_range, opt.EMBEDDING_DIM)
        self.encoder_company_name = nn.Embedding(opt.company_name_range, opt.EMBEDDING_DIM)
        self.encoder_rvcr_prov_name = nn.Embedding(opt.rvcr_prov_name_range, opt.EMBEDDING_DIM)
        self.encoder_rvcr_city_name = nn.Embedding(opt.rvcr_city_name_range, opt.EMBEDDING_DIM)
        
        self.create_day = nn.Embedding(opt.day_range, opt.EMBEDDING_DIM)
        self.create_hour = nn.Embedding(opt.hour_range, opt.EMBEDDING_DIM)
        self.pay_day = nn.Embedding(opt.day_range, opt.EMBEDDING_DIM)
        self.pay_hour = nn.Embedding(opt.hour_range, opt.EMBEDDING_DIM)

        
        
        self.FC_1_1_1 = nn.Sequential(
            nn.Linear(opt.EMBEDDING_NUM * opt.EMBEDDING_DIM, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.FC_1_1_2 = nn.Sequential(
            nn.Linear(400, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(600, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.FC_1_1_3 = nn.Sequential(
            nn.Linear(400, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(600, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.FC_1_1_4 = nn.Sequential(
            nn.Linear(400, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(600, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.FC_1_1_5 = nn.Sequential(
            nn.Linear(400, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(600, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.FC_1_1_6 = nn.Sequential(
            nn.Linear(400, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(600, opt.OUTPUT_DIM)
        )
        
        
        self.FC_1_2_1 = nn.Sequential(
            nn.Linear(opt.EMBEDDING_NUM * opt.EMBEDDING_DIM, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.FC_1_2_2 = nn.Sequential(
            nn.Linear(400, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(600, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.FC_1_2_3 = nn.Sequential(
            nn.Linear(400, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(600, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.FC_1_2_4 = nn.Sequential(
            nn.Linear(400, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(600, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.FC_1_2_5 = nn.Sequential(
            nn.Linear(400, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(600, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.FC_1_2_6 = nn.Sequential(
            nn.Linear(400, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(600, opt.OUTPUT_DIM)
        )
        

    def forward(self, x):

        '''
        embedding layers
        '''
        output_encoder_plat_form = self.encoder_plat_form(x[:,1].long())
        output_encoder_biz_type = self.encoder_biz_type(x[:,2].long())
        output_encoder_product_id = self.encoder_product_id(x[:,3].long())

        output_encoder_cate1_id = self.encoder_cate1_id(x[:,4].long())
        output_encoder_cate2_id = self.encoder_cate2_id(x[:,5].long())
        output_encoder_cate3_id = self.encoder_cate3_id(x[:,6].long())
        output_encoder_seller_uid = self.encoder_seller_uid(x[:,7].long())

        output_encoder_company_name = self.encoder_company_name(x[:,8].long())
        output_encoder_rvcr_prov_name = self.encoder_rvcr_prov_name(x[:,9].long())
        output_encoder_rvcr_city_name = self.encoder_rvcr_city_name(x[:,10].long())
        
        output_encoder_create_day = self.create_day(x[:,11].long())
        output_encoder_create_hour = self.create_hour(x[:,12].long())
        output_encoder_pay_day = self.pay_day(x[:,13].long())
        output_encoder_pay_hour = self.pay_hour(x[:,14].long())

        concat_encoder_output = torch.cat((output_encoder_plat_form, 
        output_encoder_biz_type, output_encoder_product_id, 
        output_encoder_cate1_id, output_encoder_cate2_id,
        output_encoder_cate3_id, output_encoder_seller_uid,
        output_encoder_company_name, output_encoder_rvcr_prov_name,
        output_encoder_rvcr_city_name,output_encoder_create_day,
        output_encoder_create_hour,output_encoder_pay_day,
        output_encoder_pay_hour                                   
        ), 1)

        '''
        Fully Connected layers
        you can attempt muti-task through uncommenting the following code and modifying related code in train()
        '''
        output_FC_1_1_1 = self.FC_1_1_1(concat_encoder_output)
        output_FC_1_1_2 = self.FC_1_1_2(output_FC_1_1_1)
        output_FC_1_1_3 = self.FC_1_1_3(output_FC_1_1_1 + output_FC_1_1_2)
        output_FC_1_1_4 = self.FC_1_1_4(output_FC_1_1_2 + output_FC_1_1_3)
        output_FC_1_1_5 = self.FC_1_1_5(output_FC_1_1_3 + output_FC_1_1_4)
        output_FC_1_1_6 = self.FC_1_1_6(output_FC_1_1_4 + output_FC_1_1_5)
        
        
        output_FC_1_2_1 = self.FC_1_2_1(concat_encoder_output)
        output_FC_1_2_2 = self.FC_1_2_2(output_FC_1_2_1)
        output_FC_1_2_3 = self.FC_1_2_3(output_FC_1_2_1 + output_FC_1_2_2)
        output_FC_1_2_4 = self.FC_1_2_4(output_FC_1_2_2 + output_FC_1_2_3)
        output_FC_1_2_5 = self.FC_1_2_5(output_FC_1_2_3 + output_FC_1_2_4)
        output_FC_1_2_6 = self.FC_1_2_6(output_FC_1_2_4 + output_FC_1_2_5)
        


        return (output_FC_1_1_6, None, None, None, output_FC_1_2_6, None, None, None)
