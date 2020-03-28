import time
import subprocess
import pandas as pd
import os
from sklearn.utils import shuffle

def copy_data(read_path, save_path, number):
    df = pd.read_csv(read_path)
    df_copy = df.copy(deep=True)
    origin_count = list(df_copy["label"].value_counts())[0]
    # print(list(df_copy["label"].value_counts()),origin_count)
    if origin_count < number:
        copy_num = number // origin_count
        # print("copy_num", copy_num)
        for i in range(copy_num-1):
            copied_data = df_copy.copy(deep=True)
            df = pd.concat([df, copied_data], axis=0)
    # print(list(df["label"].value_counts()))
    # after copy
    num = number - list(df["label"].value_counts())[0]
    # print(num)
    if 0 < num < origin_count:
        df_copy = shuffle(df_copy)
        labels = list(df_copy["label"].value_counts().index)
        for label in labels:
            df_copy_label = df_copy[df_copy["label"]==label][:num]
            df = pd.concat([df, df_copy_label], axis=0)

    df = shuffle(df)
    df.to_csv(save_path, index=False)

def augument_data(original_data, da_number):
    """
    :param original_data: a list of orginal data paths    type:list
    :param number: the amount of the data after data augmentation process  type:int
    """
    time_start = time.time()
    data_kda_folders = [path+'_copy_'+str(da_number)+'/' for path in original_data]

    for index, folder_p in enumerate(data_kda_folders):
        # print(index,'   ',folder_p)
        subprocess.getstatusoutput('rm -rf ' + folder_p)
        subprocess.getstatusoutput('cp -rf ' + original_data[index] + ' ' + folder_p)

        for root, dirs_p, files in os.walk(folder_p):
            for dir in dirs_p:
                for root, dirs, files in os.walk(folder_p + dir):
                    copy_data(folder_p + dir + '/train.csv', folder_p + dir + '/t',da_number)
                    break
        subprocess.getstatusoutput('find ' + folder_p + ' -name train.csv |xargs rm')
        subprocess.getstatusoutput('find ' + folder_p +'. -name t | xargs -i mv {} {}rain.csv')

    print("Augmenting all the data, take times :" + str((time.time() - time_start) / 60) + ' mins')


if __name__=="__main__":
    # To augment 5000 data
    augument_data(['data/data_500', 'data/data_2000'], 5000)

    # To augment 10000 data
    augument_data(['data/data_500', 'data/data_2000'], 10000)

