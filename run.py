# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/10/29 0:31
@File: run.py
@Desc: 参考：https://github.com/649453932/Chinese-Text-Classification-Pytorch
"""
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
import torch
import numpy as np
from train import train
from importlib import import_module
from dataset import MyDataset, build_vocab, dataset_collect
from torch.utils.data import DataLoader
import pickle as pkl
from utils import get_time_dif


if __name__ == '__main__':

    model_name = 'bert'  # transformer, bert

    x = import_module('models.' + model_name)
    config = x.Config()
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...", flush=True)
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}", flush=True)
    train_dataset = MyDataset(config.train_path, vocab, config)
    test_dataset = MyDataset(config.test_path, vocab, config)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=dataset_collect, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=dataset_collect, shuffle=False)
    time_dif = get_time_dif(start_time)
    print("Train Dataset: {}, Dataloader: {}, Test Dataset: {}, Dataloader: {}, Time usage:".format(
        len(train_dataset), len(train_dataloader), len(test_dataset), len(test_dataloader), time_dif), flush=True)
    # train
    config.n_vocab = len(vocab)
    bert_config = None
    if model_name == 'bert':
        from transformers import BertConfig
        bert_config = BertConfig(
            vocab_size=len(vocab),
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            max_position_embeddings=config.max_position_embeddings
        )
        with open(os.path.join(config.output_dir, 'config.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(vars(bert_config), indent=2, ensure_ascii=False))
    model = x.Model(**{'config': config, 'bert_config': bert_config})

    # print(model.parameters, flush=True)
    print('Begin Train ...', flush=True)
    train(config, model, train_dataloader, test_dataloader)
