import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from util import *
import math
import numpy as np
import datetime


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def calculateAllMetrics(real_signed_time_array, pred_signed_time_array):
    if len(real_signed_time_array) != len(pred_signed_time_array):
        print("[Error!] in calculateAllMetrics: len(real_signed_time_array) != len(pred_signed_time_array)")
        return -1

    score_accumulate = 0
    onTime_count = 0
    correct_count = 0
    total_count = len(real_signed_time_array)

    for i in range(total_count):
        real_signed_time = datetime.datetime.strptime(real_signed_time_array[i], "%Y-%m-%d %H:%M:%S")
        real_signed_time = real_signed_time.replace(minute = 0)
        real_signed_time = real_signed_time.replace(second = 0)
        pred_signed_time = datetime.datetime.strptime(pred_signed_time_array[i], "%Y-%m-%d %H")
        time_interval = int((real_signed_time - pred_signed_time).total_seconds() / 3600)

        # rankScore
        score_accumulate += time_interval**2

        # onTimePercent
        if pred_signed_time.year < 2019:
            onTime_count += 1
        elif pred_signed_time.year == 2019:   
            if pred_signed_time.month < real_signed_time.month:
                onTime_count += 1
            elif pred_signed_time.month == real_signed_time.month:
                if pred_signed_time.day <= real_signed_time.day:
                    onTime_count += 1

        # accuracy
        if real_signed_time.year == pred_signed_time.year and real_signed_time.month == pred_signed_time.month and real_signed_time.day == pred_signed_time.day:
            correct_count+=1

    accuracy = float(correct_count/total_count)
    onTimePercent = float(onTime_count/total_count)
    rankScore = float((score_accumulate/total_count)**0.5)

    return (rankScore,onTimePercent,accuracy)



class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False,opt = None):
        super(ArcMarginProduct, self).__init__()
        self.opt = opt
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
#         one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros_like(cosine)
#         print(one_hot.shape)
#         print(label)
        one_hot.scatter_(1, label.view(-1, 1).long().cuda(self.opt.DEVICE_ID[0]), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output