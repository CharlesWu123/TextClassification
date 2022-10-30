# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/10/29 21:46
@File: __init__.py.py
@Desc: 
"""
import os
import time


class BaseConfig(object):
    """配置参数"""

    def __init__(self, train=True):
        time_str = time.strftime("%Y%m%d-%H%M")
        output_dir = f'./output/{time_str}'
        if train:
            os.makedirs(output_dir, exist_ok=True)

        self.model_name = 'transformer'
        self.train_path = './data/train_re.csv'  # 训练集
        self.test_path = './data/test_re.csv'  # 测试集
        self.class_list = {'科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4, '社会': 5, '教育': 6, '财经': 7, '家居': 8, '游戏': 9, '房产': 10, '时尚': 11, '彩票': 12, '星座': 13}
        self.output_dir = output_dir
        self.vocab_path = './data/vocab.pkl'                                # 词表
        self.n_vocab = 0                                                # 词表大小，在运行时赋值

        self.require_improvement = 2000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 10  # epoch数
        self.batch_size = 64  # mini-batch大小
        self.pad_size = 512  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率

        self.log_iter = 100
        self.warmup = True
        self.warmup_epoch = 1