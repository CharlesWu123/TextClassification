# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/10/30 13:56
@File: dataset.py
@Desc: 
"""
import os
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset


def build_vocab(self, file_path):
    df = pd.read_csv(file_path, sep='\t')
    all_lines = ' '.join(list(df['text']))
    word_count = Counter(all_lines.split(' '))
    word_count = sorted(word_count.items(), key=lambda d: d[1], reverse=True)
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(word_count)}
    vocab_dic.update({self.UNK: len(vocab_dic), self.PAD: len(vocab_dic) + 1})
    return vocab_dic


def dataset_collect(batch):
    words, masks, labels, lens = [], [], [], []
    for word, mask, label, l in batch:
        words.append(word)
        masks.append(mask)
        labels.append(label)
        lens.append(l)
    words = torch.LongTensor(words)
    masks = torch.LongTensor(masks)
    labels = torch.LongTensor(labels)
    lens = torch.LongTensor(lens)
    return words, masks, labels, lens


class MyDataset(Dataset):
    def __init__(self, file_path, vocab, config):
        super(MyDataset, self).__init__()
        self.UNK, self.PAD = '<UNK>', '<PAD>'
        self.pad_size = config.pad_size
        self.vocab = vocab
        self.data = self.load_data(file_path)

    def load_data(self, file_path):
        contents = []
        df = pd.read_csv(file_path, sep='\t')
        for i in range(len(df)):
            words = []
            mask = []
            content, label = df.loc[i]['text'], df.loc[i]['label']
            token = content.split(' ')
            seq_len = len(token)
            mask = [1] * seq_len
            if self.pad_size:
                if seq_len < self.pad_size:
                    token.extend([self.PAD] * (self.pad_size - seq_len))
                    mask.extend([0] * (self.pad_size - seq_len))
                else:
                    token = token[:self.pad_size]
                    mask = mask[:self.pad_size]
                    seq_len = self.pad_size
            for word in token:
                words.append(self.vocab.get(word, self.vocab.get(self.UNK)))
            # words_tensor = torch.LongTensor(words)
            # mask_tensor = torch.LongTensor(mask)
            # label_tensor = torch.LongTensor(label)
            # seq_len_tensor = torch.LongTensor(seq_len)
            contents.append((words, mask, label, seq_len))
        return contents

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    import pickle as pkl
    file_path = './data/train_set.csv'
    save_path = './data/vocab.pkl'
    vocab = build_vocab(file_path=file_path)
    pkl.dump(vocab, open(save_path, 'wb'))

    # vocab = pkl.load(open(file_path, 'rb'))
    # print(vocab)