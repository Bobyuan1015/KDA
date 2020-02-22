#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
将文本整合到 train、test、val 三个文件中
"""

import numpy as np
import os
import pandas as pd
from helper.cut import split_data_by_class, func_timer


def _read_file(filename):
    """读取一个文件并转换为一行"""
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().replace('\n', '').replace('\t', '').replace('\u3000', '')

def save_file(dirname):
    """
    将多个文件整合并存到3个文件中
    dirname: 原数据目录
    文件内容格式:  类别\t内容
    """
    f_train = open('data/cnews/cnews.train.txt', 'w', encoding='utf-8')
    f_test = open('data/cnews/cnews.test.txt', 'w', encoding='utf-8')
    f_val = open('data/cnews/cnews.val.txt', 'w', encoding='utf-8')
    for category in os.listdir(dirname):   # 分类目录
        cat_dir = os.path.join(dirname, category)
        if not os.path.isdir(cat_dir):
            continue
        files = os.listdir(cat_dir)
        count = 0
        for cur_file in files:
            filename = os.path.join(cat_dir, cur_file)
            content = _read_file(filename)
            if count < 5000:
                f_train.write(category + '\t' + content + '\n')
            elif count < 6000:
                f_test.write(category + '\t' + content + '\n')
            else:
                f_val.write(category + '\t' + content + '\n')
            count += 1

        print('Finished:', category)

    f_train.close()
    f_test.close()
    f_val.close()



def train_validate_test_split(df, train_percent=.8, validate_percent=.1, seed=None):
    """
        split the dataset into train, validation, and test set
    """
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test


@func_timer
def build_train_dev_test_set(files, dir):
    """
        split the dataset into train, validation, and test set. Each category of the data has the equal sieze
    """
    max_length_of_data_set = 9999999
    train_dfs = []
    dev_dfs = []
    test_dfs = []
    for file in files:
        row_length = len(open(file, 'rU').readlines())
        if max_length_of_data_set > row_length:
            max_length_of_data_set = row_length

    for file in files:
        df = pd.read_csv(file, nrows=max_length_of_data_set)
        train_df, dev_df, test_df = train_validate_test_split(df)
        train_dfs.append(train_df)
        dev_dfs.append(dev_df)
        test_dfs.append(test_df)

    train = pd.concat(train_dfs, axis=0, ignore_index=True)
    train = train.rename(columns={train.columns[0]: 'label', train.columns[1]: 'content'})
    train.to_csv(dir+'/train.csv', index=False)
    print(dir+'train.csv len=',len(train))
    dev = pd.concat(dev_dfs, axis=0, ignore_index=True)
    dev = dev.rename(columns={dev.columns[0]: 'label', dev.columns[1]: 'content'})
    dev.to_csv(dir+'/dev.csv', index=False)
    print(dir+'dev.csv len=', len(dev))
    test = pd.concat(test_dfs, axis=0, ignore_index=True)
    test = dev.rename(columns={test.columns[0]: 'label', test.columns[1]: 'content'})
    test.to_csv(dir+'/test.csv', index=False)
    print(dir+'test.csv len=', len(test))

@func_timer
def preprocess_cnews():
    preprocessed_dir = 'data/data_orginal/cnews_10'
    data_files = ['data/data_orginal/cnews_10/original/cnews.train.txt',
                  'data/data_orginal/cnews_10/original/cnews.test.txt',
                  'data/data_orginal/cnews_10/original/cnews.val.txt']
    dataset = [preprocessed_dir+'/train.csv',
               preprocessed_dir+'/test.csv',
               preprocessed_dir+'/dev.csv']

    for index, file in enumerate(data_files):
        df = pd.read_csv(file, lineterminator='\n', delimiter='\t')
        df = df.rename(columns={df.columns[0]: 'label', df.columns[1]: 'content'})
        df.to_csv(dataset[index], index=False)
        print(dataset[index], ' size=', len(df))

def preprocess_chnsenticorp():
    preprocessed_dir = 'data/data_orginal/chnsenticorp'
    data_files = ['data/data_orginal/chnsenticorp/original/train.tsv',
                  'data/data_orginal/chnsenticorp/original/test.tsv',
                  'data/data_orginal/chnsenticorp/original/dev.tsv']
    dataset = [preprocessed_dir+'/train.csv',
               preprocessed_dir+'/test.csv',
               preprocessed_dir+'/dev.csv']

    for index, file in enumerate(data_files):
        df = pd.read_csv(file, lineterminator='\n', delimiter='\t')
        df = df.rename(columns={df.columns[0]: 'label', df.columns[1]: 'content'})

        df.to_csv(dataset[index], index=False)
        print(dataset[index],' size=',len(df))

@func_timer
def preprocess_weibo():
    data_file = 'data/data_orginal/weibo_senti_100k/original/weibo_senti_100k.csv'
    preprocessed_dir = 'data/data_orginal/weibo_senti_100k/'
    category_files = split_data_by_class(data_file, preprocessed_dir,5000)
    build_train_dev_test_set(category_files, preprocessed_dir)


if __name__ == '__main__':
    save_file('data/thucnews')
    print(len(open('data/cnews/cnews.train.txt', 'r', encoding='utf-8').readlines()))
    print(len(open('data/cnews/cnews.test.txt', 'r', encoding='utf-8').readlines()))
    print(len(open('data/cnews/cnews.val.txt', 'r', encoding='utf-8').readlines()))
