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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
import pickle as pkl
from utils import get_time_dif, WarmupPolyLR, setup_logger
from tensorboardX import SummaryWriter


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, test_iter):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
    model.to(device)
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.warmup:
        warmup_iters = config.warmup_epoch * len(train_iter)
        scheduler = WarmupPolyLR(optimizer, max_iters=config.num_epochs * len(train_iter), warmup_iters=warmup_iters,
                                 warmup_epoch=config.warmup_epoch, last_epoch=-1)
    total_batch = 0  # 记录进行到多少batch
    test_best_f1 = 0
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    writer = SummaryWriter(log_dir=config.output_dir)
    logger = setup_logger(os.path.join(config.output_dir, 'train.log'))
    model_save_dir = os.path.join(config.output_dir, 'checkpoint')
    os.makedirs(model_save_dir, exist_ok=True)
    logger.info(json.dumps(vars(config), ensure_ascii=False, indent=2))
    for epoch in range(config.num_epochs):
        lr = optimizer.param_groups[0]['lr']
        for i, batch in enumerate(train_iter):
            lr = optimizer.param_groups[0]['lr']
            inputs, masks, labels, seq_len = [x.to(device) for x in batch]
            # trains = [train.to(device) for train in trains]
            # labels = labels.to(device)
            outputs = model(inputs, masks)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if config.warmup:
                scheduler.step()
            if total_batch % config.log_iter == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predict = torch.max(outputs.data, 1)[1].cpu()
                # train_acc = metrics.accuracy_score(true, predict)
                train_acc, train_rec, train_f1, _ = metrics.precision_recall_fscore_support(true, predict, average='macro', zero_division=0)
                test_acc, test_rec, test_f1, test_loss = evaluate(config, model, test_iter)
                if test_f1 > test_best_f1:
                    test_best_f1 = test_f1
                    torch.save(model.state_dict(), os.path.join(model_save_dir, f'best_{epoch}.ckpt'))
                    torch.save(model.state_dict(), os.path.join(model_save_dir, f'best.ckpt'))
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Epoch [{}/{}], Iter: {:>6},  Train Loss: {:>5.2f},  Train F1: {:>6.2%},  Test Loss: {:>5.2f}, Test F1: {:>6.2%}, LR: {:>7.6f},Time: {} {}'
                logger.info(msg.format(epoch + 1, config.num_epochs, total_batch, loss.item(), train_f1, test_loss, test_f1, lr, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/test", test_loss, total_batch)
                writer.add_scalar("f1/train", train_f1, total_batch)
                writer.add_scalar("f1/test", test_f1, total_batch)
                writer.add_scalar("train/lr", lr, total_batch)

                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                logger.info("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    test(config, model, test_iter, logger, os.path.join(model_save_dir, f'best.ckpt'))


def test(config, model, test_iter, logger, best_model_path):
    # test
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    start_time = time.time()
    test_acc, test_recall, test_f1, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}, Test Rec: {2:>6.2%}, Test F1: {3:>6.2%}'
    logger.info(msg.format(test_loss, test_acc, test_recall, test_f1))
    logger.info("\nPrecision, Recall and F1-Score...")
    logger.info(test_report)
    logger.info("Confusion Matrix...\n")
    logger.info('\n{}'.join(test_confusion))
    time_dif = get_time_dif(start_time)
    logger.info("Time usage:".format(time_dif))


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for batch in data_iter:
            inputs, masks, labels, seq_len = [x.cuda() for x in batch]
            # texts = [text.cuda() for text in texts]
            # labels = labels.cuda()
            outputs = model(inputs, masks)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)

    # acc = metrics.accuracy_score(labels_all, predict_all)
    acc, recall, f1, _ = metrics.precision_recall_fscore_support(labels_all, predict_all, average='macro', zero_division=0)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, recall, f1, loss_total / len(data_iter), report, confusion
    return acc, recall, f1, loss_total / len(data_iter)