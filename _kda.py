import os
import random
import re
import time
from collections import defaultdict
import pandas as pd
from sklearn.utils import shuffle
import subprocess
import math

def augment(replace_sent: str, to_replace_list: list, synonym_dict: dict,rest_row_num,rest_da_num):
    da_sentences = []
    changed_sentence = replace_sent
    da_num = math.ceil(rest_da_num/rest_row_num)
    print(da_num, " replace_sent:", replace_sent)
    if da_num > 0:
        for i in range(da_num):
            for index, replace_word in enumerate(to_replace_list):
                if re.search(replace_word, replace_sent) and i < len(synonym_dict[replace_word]):
                    # if "服装品牌" == replace_word and "组图：成熟美丽阿娇的华丽转身经历了艳照门" in replace_sent:
                    changed_sentence = re.sub(replace_word, synonym_dict[replace_word][i], replace_sent, 1)
            da_sentences.append(changed_sentence)
    # print(len(da_sentences),"da_sentences: ", da_sentences)
    for sen in da_sentences:
        print(sen)
    return da_sentences


def old_augment(replace_sent: str, to_replace_list: list, synonym_dict: dict):
    """
    :param replace_sent: the senences to be replaced
    :param to_replace_list: the words to be replaced in the senences
    :param synonym_dict: a synonym dictionary.
    :return:str
    """
    da_sentences = []

    for replace_word in to_replace_list:
        if re.search(replace_word, replace_sent):
            for synonym_word in random.shuffle(synonym_dict[replace_word]):
                da_sentences.append(re.sub(replace_word, synonym_word, replace_sent, 1))

    # da_sentences = []
    # for replace_word in to_replace_list:
    #     if re.search(replace_word, replace_sent):
    #         for synonym_word in synonym_dict[replace_word]:
    #             da_sentences.append(re.sub(replace_word, synonym_word, replace_sent, 1))
    return da_sentences

def search_replacement(sentence, words):
    searched_list = []
    for word in words:
        if word in sentence:
            searched_list.append(word)
    return searched_list

def get_worked_keys(dir):
    key_files_name = []
    keys_a_file = []
    for root, dirs, files in os.walk(dir):
        for file_name in files:
            if 'keys.csv' in file_name:
                key_files_name.append(dir+'/'+file_name)
                df = pd.read_csv(dir +'/'+ file_name)
                keys_a_file.append(df['keys'].tolist())

    print(key_files_name)
    def chaji_keys(set1, setList):
        set1 = set(set1)
        for a_list in setList:
            print('0 set1=', len(set1), len(set(a_list)))
            set1 -= set(a_list)
            print('1 set1=', len(set1))
        print('---------')
        return list(set1)


    def jiaoji_keys(set1, setList):
        set1 = set(set1)

        for a_list in setList:
            print('0 set1=', len(set1), len(set(a_list)))
            set1 &= set(a_list)
            print('1 set1=', len(set1))
        print('---------')
        return list(set1)

    for i, key_a_file in enumerate(keys_a_file):
        temp_list = keys_a_file.copy()
        temp_list.pop(i)
        keys_work = jiaoji_keys(key_a_file, temp_list)
        df = pd.DataFrame({'keys': keys_work})

        df.to_csv( key_files_name[i].split('.csv')[0] + str(len(keys_work))+"_work.csv", index=False)
        print(key_files_name[i].split('.csv')[0] + str(len(keys_work))+"_work.csv")


def match_keys(file_path,dict_path):
    df = pd.read_csv(file_path + '.csv')
    row_num = len(df)
    print('kda open ', file_path, row_num)
    df_synonyms = pd.read_csv(dict_path)
    df_synonyms.drop(df_synonyms[df_synonyms.final_all_keys.isnull()].index, inplace=True)
    df_synonyms.drop(df_synonyms[df_synonyms.close_words.isnull()].index, inplace=True)
    synonym_dict = defaultdict(list)
    for index, row in df_synonyms.iterrows():
        synonym_dict[row['final_all_keys']] = row['close_words'].split(',')
    hit_keywords = []
    for index, row in df.iterrows():
        sentence = row['content']
        keywords_to_be_replaced = search_replacement(sentence, df_synonyms['final_all_keys'].tolist())
        hit_keywords.extend(keywords_to_be_replaced)

    matched_keys_df = pd.DataFrame({'keys':hit_keywords})
    matched_keys_df.to_csv(file_path + '_keys.csv', index=False)
    print('save ', file_path + '_keys.csv', len(matched_keys_df))

def kda(file_path, dict_path, newsize=-1):

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


def kda_single_file(dir, file_path, dict_path, newsize):

    df = pd.read_csv(file_path+'.csv')
    row_num = len(df)
    print('kda open ',file_path,row_num )
    df_synonyms = pd.read_csv(dict_path)
    df_synonyms.drop(df_synonyms[df_synonyms.final_all_keys.isnull()].index, inplace=True)
    df_synonyms.drop(df_synonyms[df_synonyms.close_words.isnull()].index, inplace=True)
    synonym_dict = defaultdict(list)
    for index, row in df_synonyms.iterrows():
        synonym_dict[row['final_all_keys']] = row['close_words'].split(',')

    da_sentences = []
    da_sentences_number = []
    da_total = 0
    hit_keywords = []
    for index, row in df.iterrows():
        sentence = row['content']
        keywords_to_be_replaced = search_replacement(sentence, df_synonyms['final_all_keys'].tolist())
        hit_keywords.extend(keywords_to_be_replaced)
        new_sents = augment(sentence, keywords_to_be_replaced, synonym_dict)
        da_sentences.append(new_sents)
        da_total += len(new_sents)
        da_sentences_number.append(len(new_sents))

    df1 = df_synonyms[df_synonyms.final_all_keys.isin(set(hit_keywords))]
    df1.to_csv(file_path + '_keys.csv', index=False)
    print('save ',file_path + '_keys.csv',len(df1))

    df['da_sentences'] = pd.DataFrame({'da_sentences': da_sentences})
    df['da_sentences_number'] = pd.DataFrame({'da_sentences_number': da_sentences_number})
    # df.to_csv(file_path+'_daDebug.csv', index=False) #for debugging
    selected_precentage = (newsize - row_num) / da_total

    new_sents = []
    new_labels = []
    rest_number = newsize - row_num
    for index, row in df.iterrows():
        new_sents.append(row['content'])#the original sentence
        new_labels.append(row['label'])
        selected_number = int(row['da_sentences_number']*selected_precentage)
        if selected_number < 15 and selected_number>0:
            selected_number += 1

        if rest_number <= selected_number:
            # #print('rest_number=', rest_number, ' selected_number=', selected_number)
            selected_number = rest_number
        for i in range(selected_number):#the generated sentences
            # #print('i=',i,' len(row[da_sentences]=',len(row['da_sentences']))
            new_sents.append(row['da_sentences'][i])
            new_labels.append(row['label'])
        rest_number -= selected_number

    #print('rest_number=', rest_number)
    da_df = pd.DataFrame({'label':new_labels, 'content':new_sents})
    da_df.to_csv(file_path+'_kda.csv', index=False)
    return file_path+'_kda.csv' ,

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
    concat_df = shuffle(concat_df)
    concat_df.to_csv(merge_path, index=False)
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


def augument_data(original_data, da_number):
    """
    :param original_data: a list of orginal data paths    type:list
    :param number: the amount of the data after data augmentation process  type:int
    """
    data_kda_folders = [path+'_kda_'+str(da_number)+'/' for path in original_data]
    refined_keywords_dir = 'data/refine_keywords/'
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
                        if 'test' not in file_name and 'dev' not in file_name and 'keys' not in file_name:
                            # match_keys( folder_p + dir + '/' + file_name.split('.csv')[0], dict_path )
                            f_name = file_name.split('.csv')[0]
                            dict_path = refined_keywords_dir + dir + '/' + f_name + 'refine.csv'
                            kda(folder_p + dir + '/' + f_name, dict_path, da_number)
                            # return

    print("Augmenting all the data, take times :" + str((time.time() - time_start) / 60) + ' mins')
    # build training set
    for folder_p in data_kda_folders:
        for root, dirs_p, files in os.walk(folder_p):
            for dir in dirs_p:
                merge_path(folder_p + dir, 'train.csv', '_kda.csv')
            break

if __name__ == "__main__":
    #To augment 5000 data
    augument_data(['data/data_500', 'data/data_2000'],5000)
    augument_data(['data/data_2000', 'data/data_2000'], 5000)

    # To augment 10000 data
    #augument_data(['data/data_500', 'data/data_2000'], 10000)
