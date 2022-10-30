# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/10/29 16:07
@File: gen_test_res.py
@Desc: 
"""
import os
import time

import pandas as pd
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from importlib import import_module
import pickle as pkl


UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


class Predict:
    def __init__(self, model_path, model_name='bert'):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        x = import_module('models.' + model_name)
        config = x.Config(train=False)
        self.pad_size = config.pad_size
        self.vocab = pkl.load(open(config.vocab_path, 'rb'))
        config.n_vocab = len(self.vocab)
        bert_config = None
        if model_name == 'bert':
            from transformers import BertConfig
            bert_config = BertConfig(
                vocab_size=config.n_vocab,
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                hidden_dropout_prob=config.hidden_dropout_prob,
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                max_position_embeddings=config.max_position_embeddings
            )
        self.model = x.Model(**{'config': config, 'bert_config': bert_config})
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

    def trans(self, texts):
        datas = []
        for content in texts:
            words_line = []
            mask = []
            token = content.split(' ')
            seq_len = len(token)
            mask = [1] * seq_len
            if self.pad_size:
                if len(token) < self.pad_size:
                    token.extend([PAD] * (self.pad_size - len(token)))
                    mask.extend([0] * (self.pad_size - seq_len))
                else:
                    token = token[:self.pad_size]
                    mask = mask[:self.pad_size]
                    seq_len = self.pad_size
            # word to id
            for word in token:
                words_line.append(self.vocab.get(word, self.vocab.get(UNK)))
            datas.append((words_line, mask, seq_len))

        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        return x, mask, seq_len

    def __call__(self, texts):
        inputs, masks, seq_len = self.trans(texts)
        outputs = self.model(inputs, masks)
        predict = torch.max(outputs.data, 1)[1].cpu().numpy().tolist()
        return predict


def test(file_path, model):
    pred = []
    df = pd.read_csv(file_path, sep='\t')
    for text in tqdm(list(df['text'])):
        output = model([text])
        pred.append(output[0])
    write_data = ['label']
    write_data += map(str, pred)
    with open(f'./data/test_res_{time.strftime("%Y%m%d-%H%M")}.csv', 'w', encoding='utf-8') as f:
        f.write('\n'.join(write_data))


if __name__ == '__main__':
    model_name = 'bert'
    model_path = './output/20221030-1633/checkpoint/best.ckpt'
    model = Predict(model_path, model_name)
    test('./data/test_a.csv', model)


