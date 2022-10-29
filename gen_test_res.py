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
from models.transformer import Config, Model
import pickle as pkl


UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


class Predict:
    def __init__(self, model_path, config):
        self.pad_size = config.pad_size
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.vocab = pkl.load(open(config.vocab_path, 'rb'))
        config.n_vocab = len(self.vocab)
        self.model = Model(config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

    def trans(self, texts):
        datas = []
        for content in texts:
            words_line = []
            token = content.split(' ')
            seq_len = len(token)
            if self.pad_size:
                if len(token) < self.pad_size:
                    token.extend([PAD] * (self.pad_size - len(token)))
                else:
                    token = token[:self.pad_size]
                    seq_len = self.pad_size
            # word to id
            for word in token:
                words_line.append(self.vocab.get(word, self.vocab.get(UNK)))
            datas.append((words_line, seq_len))

        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        return x, seq_len

    def __call__(self, texts):
        inputs = self.trans(texts)
        outputs = self.model(inputs)
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
    dataset = ''
    embedding = 'random'
    model_path = './output/20221029-1930/Transformer.ckpt'
    config = Config(dataset, embedding, train=False)
    model = Predict(model_path, config)
    test('./data/test_a.csv', model)


