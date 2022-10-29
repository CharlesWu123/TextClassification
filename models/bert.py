# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/10/29 21:45
@File: bert.py
@Desc: 
"""
# coding: UTF-8
import os
import time

import torch
import torch.nn as nn
from transformers import BertModel


class Config(object):

    """配置参数"""
    def __init__(self, dataset, train=True):
        time_str = time.strftime("%Y%m%d-%H%M")
        output_dir = f'./output/{time_str}'
        if train:
            os.makedirs(output_dir, exist_ok=True)

        self.model_name = 'bert'
        self.train_path = './data/train_re.csv'     # 训练集
        self.test_path = './data/test_re.csv'       # 测试集
        self.class_list = {'科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4, '社会': 5, '教育': 6, '财经': 7, '家居': 8, '游戏': 9, '房产': 10, '时尚': 11, '彩票': 12, '星座': 13}
        self.output_dir = output_dir

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 10                                             # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 512                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = './bert_pretrain'
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config, bert_config):
        super(Model, self).__init__()
        self.bert = BertModel(bert_config)
        # self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out