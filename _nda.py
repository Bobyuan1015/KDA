# -*- coding: utf-8 -*-
"""
   File Name：     _nda.py
   Description :  data augmentation by extending noise data into data set.
   Author :       yuanfang
   date：         2019/12/13
"""

# os模块中包含很多操作文件和目录的函数  
import os
import subprocess
import time
from collections import defaultdict
import pandas as pd
import jieba
import re
from collections import Counter
import collections
from sklearn.utils import shuffle

def merge_csv(files, merge_path):
    if len(files) < 1:
        return None
    dfs = []
    for f in files:
        if f is None:
            return None
        else:
            df = pd.read_csv(f)
            df = shuffle(df)
            dfs.append(df)
    concat_df = pd.concat(dfs, axis=0, ignore_index=True)
    concat_df = concat_df.loc[:, ['final_all_keys']]
    concat_df = shuffle(concat_df)
    concat_df.drop_duplicates(subset=['final_all_keys'], keep='first', inplace=True)  # remove duplication rows
    concat_df.to_csv(merge_path, index=False)
    return len(concat_df), merge_path


def merge_all_refine_keys(refine_dir):
    print(refine_dir)
    paths = []
    for root, dirs_p, files in os.walk(refine_dir):
        for dir in dirs_p:
            for root, dirs, files in os.walk(refine_dir + dir):
                for file_name in files:

                    if file_name.endswith('refine.csv'):
                        print(f"  {file_name}")
                        paths.append(refine_dir + dir+'/'+file_name)
    print(paths)
    l,merge_path = merge_csv(paths, refine_dir+'all_refine.csv')
    print(f"merge len=", l)
    return merge_path

def merge_path(path, out_file_name, sub_name_in_file='clean'):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('max_colwidth', 100)
    # -----------merge all files
    path = path+'/'
    tomerge_files = []
    for root, dirs, files in os.walk(path):
        for file_name in files:
            if sub_name_in_file in file_name:
                tomerge_files.append(path + file_name)
        break
    #print('merge files:',tomerge_files)
    merge_csv(tomerge_files, path + out_file_name)


def mergse_all_txts(dir):
    # 获取目标文件夹的路径
    # meragefiledir = os.getcwd() + '//MerageFiles'
    meragefiledir = dir
    # 获取当前文件夹中的文件名称列表
    filenames = os.listdir(meragefiledir)
    stopwords = []
    # 先遍历文件名
    for filename in filenames:
        if '.txt' in filename:
            filepath = meragefiledir + '/'
            filepath = filepath + filename
            # 遍历单个文件，读取行数
            for line in open(filepath):
                stopwords.append(line)
    # 关闭文件
    allstops = list(set(stopwords))
    df = pd.DataFrame({'stops':allstops})
    print(f"old len={len(stopwords)},  new len={len(df)}")
    df.to_csv(dir+'/stops.csv',index=False)
    return dir+'/stops.csv'


def dataPrepos(text, stopkey):
    # l = []
    # pos = ['n','v','vn']
    #'nz',#名词
    #'v',
    # 'vd',
    #'vn',#动词
    #'l',
    #'a',#形容词
    # 'd'#副词
    #]  # 定义选取的词性
    seg = jieba.cut(text)  # 分词
    # for i in seg:
    #     if i.word not in stopkey:
    #     # if i.word not in stopkey:  # 去停用词 + 词性筛选
    #         l.append(i.word)
    l=[]
    for aseg in seg:
        if len(aseg) > 1 and ' ' not in aseg and not bool(re.search(r'\d', aseg)):
            l.append(aseg)
    return l


def compute_useless_words(file,exclude_words):
    df = pd.read_csv(file)

    df['segs'] = df.apply(lambda row:dataPrepos(row['content'],exclude_words),axis=1)
    text = []
    for alist in df['segs'].to_list():
        text.extend(alist)

    c = Counter(text)
    # all_size = len(set(list(c.elements())))
    # print("all_size=",all_size)
    count_pairs = c.most_common()
    words, _ = list(zip(*count_pairs))
    hight_tf_words = words[:5000]

    low_tf_words = words[-5000:]
    stopword_dictionary = list(set(hight_tf_words+low_tf_words))
    print(c)
    return stopword_dictionary



def get_useless(save_dir,root_dir, refine_path):

    all_files = ['1.csv', '0.csv', '体育.csv',
                 '娱乐.csv', '家居.csv', '房产.csv',
                 '教育.csv', '时尚.csv', '游戏.csv',
                 '科技.csv', '财经.csv', '时政.csv']

    root_dir = 'data/data_orginal/'
    refine_df = pd.read_csv(refine_path)
    refine_words = refine_df['final_all_keys'].to_list()
    for root, dirs_p, files in os.walk(root_dir):
        all_keys = []
        final_all_keys = set()
        for dir in dirs_p:
            if dir in ['cnews_10', 'weibo_senti_100k', 'chnsenticorp']:
                save_folder = save_dir+dir
                subprocess.getstatusoutput(f"mkdir -p {save_folder}")
                for root_, dirs_, files_ in os.walk(root_dir+dir):
                    keys_classes = []
                    for file_name in files_:
                        if file_name in all_files:
                            file = root_dir + dir + '/' + file_name
                            print(file)
                            useless_words = compute_useless_words(file, refine_words)
                            key_save_path = save_folder + '/' + file_name + '_useless_size' + str(len(useless_words)) + '.csv'
                            df = pd.DataFrame({'useless_words':useless_words})
                            df.to_csv(key_save_path, index=False)



def search_replacement(sentence, words):
    searched_list = []
    for word in words:
        if word in sentence:
            searched_list.append(word)
    return searched_list


def nda(file_path, dict_path, newsize=-1):

    df = pd.read_csv(file_path+'.csv')
    row_num = len(df)
    da_size = newsize - row_num

    print('kda open ', file_path, row_num, ' ', dict_path)
    df_synonyms = pd.read_csv(dict_path)
    df_synonyms.drop(df_synonyms[df_synonyms.final_all_keys.isnull()].index, inplace=True)
    df_synonyms.drop(df_synonyms[df_synonyms.close_words.isnull()].index, inplace=True)
    synonym_dict = defaultdict(list)
    for index, row in df_synonyms.iterrows():
        synonym_dict[row['final_all_keys']] = row['close_words'].split(',')

    da_sentences = []
    hit_keywords = []

    for index, row in df.iterrows():
        sentence = row['content']
        keywords_to_be_replaced = search_replacement(sentence, df_synonyms['final_all_keys'].tolist())
        hit_keywords.append(keywords_to_be_replaced)

    labels = []
    keys = []
    for index, row in df.iterrows():
        sentence = row['content']
        new_sents = augment(sentence, hit_keywords[index], synonym_dict, row_num-index, da_size)
        da_size -= len(new_sents)
        da_sentences.extend(new_sents)
        for i in range(len(new_sents)):
            labels.append(row['label'])
            keys.append(hit_keywords[index])
        if da_size < 1:
            break
    df_da = pd.DataFrame({"content": da_sentences, "label": labels, "keys": keys})
    new_df = pd.concat([df, df_da])
    new_df.to_csv(file_path+'_kda.csv', index=False)
    return file_path+'_kda.csv'

def augument_data(original_data, da_number):
    """
    :param original_data: a list of orginal data paths    type:list
    :param number: the amount of the data after data augmentation process  type:int
    """
    data_kda_folders = [path+'_nda_'+str(da_number)+'/' for path in original_data]
    useless_words_dir = 'data/stopwords/'

    time_start = time.time()

    for index, folder_p in enumerate(data_kda_folders):
        # print(index,'   ',folder_p)
        subprocess.getstatusoutput('rm -rf ' + folder_p)
        subprocess.getstatusoutput('cp -rf ' + original_data[index] + ' ' + folder_p)
        subprocess.getstatusoutput('find ' + folder_p + ' -name train.csv |xargs rm')

        for root, dirs_p, files in os.walk(folder_p):
            for dir in dirs_p:
                for root, dirs, files in os.walk(folder_p + dir):
                    for file_name in files:
                        if 'train.csv' not in file_name:
                            f_name = file_name.split('.csv')[0]
                            dict_path = useless_words_dir + dir + '/' + f_name + 'picked.csv' #近义词+refine的停用词
                            nda(folder_p + dir + '/' + f_name, dict_path, da_number)


    print("Augmenting all the data, take times :" + str((time.time() - time_start) / 60) + ' mins')
    # build training set
    for folder_p in data_kda_folders:
        for root, dirs_p, files in os.walk(folder_p):
            for dir in dirs_p:
                merge_path(folder_p + dir, 'train.csv', '_nda.csv')
            break




#1.获得所有refine的词
# refine_path =  merge_all_refine_keys("data/refine_keywords/")

#2.收集网络上的停用词
# stops_path = mergse_all_txts("data/stopwords")

#3.分词并计算词频
# refine_path='data/refine_keywords/all_refine.csv'
# get_useless("data/stopwords/", "data/data_orginal/", refine_path)

#4.数据增强
augument_data(['data/data_500', 'data/data_2000'],5000)