import os
import re
import time
from collections import defaultdict
import pandas as pd
import subprocess

def augment(replace_sent: str, to_replace_list: list, synonym_dict: dict):
    """
    :param replace_sent: 需要被替换的句子
    :param to_replace_list: 需要替换的词
    :param synonym_dict: 需要替换的同义词字典
    :return:str
    """
    da_sentences = []
    for replace_word in to_replace_list:
        if re.search(replace_word, replace_sent):
            for synonym_word in synonym_dict[replace_word]:
                da_sentences.append(re.sub(replace_word, synonym_word, replace_sent, 1))
    return da_sentences



def kda_old(file_path, dict_path, newsize):

    df = pd.read_csv(file_path+'.csv')
    row_num = len(df)

    df_synonyms = pd.read_csv(dict_path)
    df_synonyms.drop(df_synonyms[df_synonyms.final_all_keys.isnull()].index, inplace=True)
    df_synonyms.drop(df_synonyms[df_synonyms.close_words.isnull()].index, inplace=True)
    synonym_dict = defaultdict(list)
    for index, row in df_synonyms.iterrows():
        synonym_dict[row['final_all_keys']] = row['close_words'].split(',')

    da_sentences = []
    da_sentences_number = []
    da_total = 0
    for index, row in df.iterrows():
        sentence = row['content']
        keywords_to_be_replaced = str(row['Intersection']).split(",")
        new_sents = augment(sentence, keywords_to_be_replaced, synonym_dict)
        da_sentences.append(new_sents)
        da_total += len(new_sents)
        da_sentences_number.append(len(new_sents))
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
            # print('rest_number=', rest_number, ' selected_number=', selected_number)
            selected_number = rest_number
        for i in range(selected_number):#the generated sentences
            # print('i=',i,' len(row[da_sentences]=',len(row['da_sentences']))
            new_sents.append(row['da_sentences'][i])
            new_labels.append(row['label'])
        rest_number -= selected_number

    print('rest_number=', rest_number)
    da_df = pd.DataFrame({'label':new_labels, 'content':new_sents})
    da_df.to_csv(file_path+'_kda.csv', index=False)
    return file_path+'_kda.csv'

def search_replacement(sentence, words):
    searched_list = []
    for word in words:
        if word in sentence:
            searched_list.append(word)
    return searched_list


def kda(file_path, dict_path, newsize):
    print('kda open ',file_path)
    df = pd.read_csv(file_path+'.csv')
    row_num = len(df)

    df_synonyms = pd.read_csv(dict_path)
    df_synonyms.drop(df_synonyms[df_synonyms.final_all_keys.isnull()].index, inplace=True)
    df_synonyms.drop(df_synonyms[df_synonyms.close_words.isnull()].index, inplace=True)
    synonym_dict = defaultdict(list)
    for index, row in df_synonyms.iterrows():
        synonym_dict[row['final_all_keys']] = row['close_words'].split(',')

    da_sentences = []
    da_sentences_number = []
    da_total = 0
    for index, row in df.iterrows():
        sentence = row['content']
        keywords_to_be_replaced = search_replacement(sentence,df_synonyms['final_all_keys'].tolist())
        new_sents = augment(sentence, keywords_to_be_replaced, synonym_dict)
        da_sentences.append(new_sents)
        da_total += len(new_sents)
        da_sentences_number.append(len(new_sents))
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
            # print('rest_number=', rest_number, ' selected_number=', selected_number)
            selected_number = rest_number
        for i in range(selected_number):#the generated sentences
            # print('i=',i,' len(row[da_sentences]=',len(row['da_sentences']))
            new_sents.append(row['da_sentences'][i])
            new_labels.append(row['label'])
        rest_number -= selected_number

    print('rest_number=', rest_number)
    da_df = pd.DataFrame({'label':new_labels, 'content':new_sents})
    da_df.to_csv(file_path+'_kda.csv', index=False)
    return file_path+'_kda.csv'

def merge_csv(files, merge_path):
    # , lineterminator = '\n'))

    print('merge_csv: ',files)
    if len(files) < 1:
        print(' No one file, so it cannot merge')
        return None
    dfs = []
    for f in files:
        if f is None:
            print('file is null, then return')
            return None
        else:
            print('open ',f,)
            df = pd.read_csv(f)
            # df = pd.read_csv(f, lineterminator='\n')

            # df.drop(df[df.CONTENT.isnull()].index,
            #         inplace=True)  # could be improved if remove all redundant in 'repjection' case
            dfs.append(df)
    concat_df = pd.concat(dfs, axis=0, ignore_index=True)
    concat_df.to_csv(merge_path,index=False)
    return merge_path

def merge_path(path, out_file_name, sub_name_in_file='clean'):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('max_colwidth', 100)

    # -----------merge all files
    print('merge_path:',path)
    path = path+'/'
    tomerge_files = []
    for root, dirs, files in os.walk(path):

        for file_name in files:
            # print('file_name=',file_name)
            # for account_type in accounts:
            if sub_name_in_file in file_name:
                tomerge_files.append(path + file_name)
        break
    print('merge files:',tomerge_files)
    merge_csv(tomerge_files, path + out_file_name)

if __name__ == "__main__":

    data_kda_folders = ['data-500_kda5000/', 'data-2000_kda5000/']
    data_cut_folders = ['data-500', 'data-2000']
    dict_path = "final0129.csv"
    time_start = time.time()
    toKDA_files = []
    toMERG_files = []
    for index,folder_p in enumerate(data_kda_folders):
        print(index,'   ',folder_p)
        subprocess.getstatusoutput('rm -rf ' + folder_p)
        subprocess.getstatusoutput('cp -rf ' + data_cut_folders[index] + ' ' + folder_p)
        subprocess.getstatusoutput('find ' + folder_p + ' -name train.csv |xargs rm')
        # subprocess.getstatusoutput('find ' + folder_p + ' -name dev.csv |xargs rm')
        # subprocess.getstatusoutput('find ' + folder_p + ' -name test.csv |xargs rm')
        for root, dirs_p, files in os.walk(folder_p):
            for dir in dirs_p:
                for root, dirs, files in os.walk(folder_p+dir):
                    for file_name in files:

                        # if '_key.csv' in file_name:
                            # toKDA_files.append()
                        kda(folder_p + dir + '/' + file_name.split('.csv')[0], dict_path, 5000)
                            # toKDA_files.append(folder_p+dir + '/' + file_name)
                            # print('file_name=', file_name)

    print("Augmenting all the data 耗时:" + str((time.time() - time_start) / 60) + ' 分')
    # print('KDA files:', toKDA_files)


    # for file_path in toKDA_files:
    #     print('Precessing ', file_path)
    #     kda(file_path.split('.csv')[0], dict_path, 5000)

    #build training set
    for folder_p in data_kda_folders:
        for root, dirs_p, files in os.walk(folder_p):
            for dir in dirs_p:
                merge_path(folder_p+dir,'train.csv','_kda.csv')
            break
