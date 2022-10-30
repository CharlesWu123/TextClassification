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
from . import BaseConfig


class Config(BaseConfig):
    """配置参数"""
    def __init__(self, train=True):
        super(Config, self).__init__(train)
        self.model_name = 'bert'
        self.bert_path = './bert_pretrain'
        self.hidden_size = 768
        self.num_hidden_layers = 6
        self.num_attention_heads = 6
        self.intermediate_size = 1024
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512


class Model(nn.Module):
    def __init__(self, config, bert_config, **kwargs):
        super(Model, self).__init__()
        self.bert = BertModel(bert_config)

        # self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x, mask):
        outputs = self.bert(x, attention_mask=mask, return_dict=True)
        out = self.fc(outputs.pooler_output)
        return out