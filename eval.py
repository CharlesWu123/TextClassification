# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/10/29 0:00
@File: train.py
@Desc: 
"""
# coding: UTF-8
import json
import os

from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time

from models.transformer import Config, Model
from utils import get_time_dif, WarmupPolyLR, setup_logger, build_dataset, build_iterator
from tensorboardX import SummaryWriter


def test(config, model, test_iter, best_model_path):
    logger = setup_logger(os.path.join(config.output_dir, 'test.log'))
    logger.info(json.dumps(vars(config), ensure_ascii=False, indent=2))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
    # test
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    model.eval()
    start_time = time.time()
    test_acc, test_recall, test_f1, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}, Test Rec: {2:>6.2%}, Test F1: {3:>6.2%}'
    logger.info(msg.format(test_loss, test_acc, test_recall, test_f1))
    logger.info("\nPrecision, Recall and F1-Score...")
    logger.info(test_report)
    logger.info("\nConfusion Matrix...")
    logger.info(test_confusion)
    time_dif = get_time_dif(start_time)
    logger.info("Time usage: {}".format(time_dif))


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in tqdm(data_iter):
            texts = [text.cuda() for text in texts]
            labels = labels.cuda()
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    # acc = metrics.accuracy_score(labels_all, predict_all)
    acc, recall, f1, _ = metrics.precision_recall_fscore_support(labels_all, predict_all, average='macro')
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, recall, f1, loss_total / len(data_iter), report, confusion
    return acc, recall, f1, loss_total / len(data_iter)


if __name__ == '__main__':
    dataset = ''
    embedding = 'random'
    model_path = './output/20221029-1930/Transformer.ckpt'

    config = Config(dataset, embedding)
    print("Loading data...", flush=True)
    vocab, _, test_data = build_dataset(config, False, train=False)
    test_iter = build_iterator(test_data, config)
    print("Test...", flush=True)
    config.n_vocab = len(vocab)
    model = Model(config)
    test(config, model, test_iter, model_path)