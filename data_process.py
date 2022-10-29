# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Charles
@Time: 2022/10/28 23:30
@File: data_process.py
@Desc: 
"""
import matplotlib.pyplot as plt
import pandas as pd

def data_statics():
    # 数据分析
    # 句子长度分析
    train_df['text_len'] = train_df['text'].apply(lambda x: len(x.split(' ')))
    print(train_df['text_len'].describe())
    # 绘制直方图
    _ = plt.hist(train_df['text_len'], bins=200)
    plt.xlabel('Text char count')
    plt.title('Histogram of char count')
    plt.savefig('./output/text_char_count.png')
    # 新闻类别分布
    train_df['label'].value_counts().plot(kind='bar')
    plt.title('News class count')
    plt.xlabel('category')
    # plt.show()
    plt.savefig('./output/news_class_count.png')
    # 字符分布统计
    from collections import Counter
    all_lines = ' '.join(list(train_df['text']))
    word_count = Counter(all_lines.split(' '))
    word_count = sorted(word_count.items(), key=lambda d:d[1], reverse=True)
    print(len(word_count))      # 6869
    print(word_count[0])        # ('3750', 7482224)
    print(word_count[1])        # ('648', 4924890)
    print(word_count[2])        # ('900', 3262544)
    print(word_count[-1])       # ('3133', 1)


def split():
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(train_df, test_size=10000)
    train.to_csv('./data/train_re.csv', sep='\t', index=False)
    test.to_csv('./data/test_re.csv', sep='\t', index=False)


if __name__ == '__main__':
    # 读取数据
    train_df = pd.read_csv('./data/train_set.csv', sep='\t')
    # train_df = pd.read_csv('./data/test_a.csv', sep='\t')
    # print(train_df.head(n=10))
    # split()
    data_statics()