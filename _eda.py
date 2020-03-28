import subprocess
import time
import os
import math
import pandas as pd


def eda_data(origin_file, outfile, number):
    # original data

    # origin = "../data/data_500/cnews_10/train.csv"
    # input_file = "../data/data_500_eda5000/cnews_10/train_500_eda4500.csv"
    # outfile = "../data/data_500_eda5000/cnews_10/train_500_eda5000.csv"

    origin = pd.read_csv(origin_file)
    origin = origin[["label", "content"]]
    origin_count = list(origin["label"].value_counts())
    labels = list(origin["label"].value_counts().index)
    amount_a_class = len(origin[origin["label"] == labels[0]])
    da_times = math.ceil((number - amount_a_class) / amount_a_class)

    # eda data
    # https://github.com/zhanlaoban/EDA_NLP_for_Chinese
    # paper: 《EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks》
    os.system("python eda_chinese/augment.py --input={} --output={} --num_aug={} --alpha=0.05".format(origin_file, outfile, da_times))

    eda_data = pd.read_csv(outfile)
    eda_data = eda_data[["label", "content"]]
    label_count = list(eda_data["label"].value_counts())

    # check the summation of the number of eda data and the number of original data with the parameter (number)
    if origin_count[0] + label_count[0] < number:
        return None
    elif origin_count[0] + label_count[0] == number:
        train = pd.concat([origin, eda_data], axis=0)
    else:
        for i, label in enumerate(labels):
            pick_num = number - origin_count[i]
            picked_data = eda_data[eda_data["label"] == label][:pick_num]
            origin = pd.concat([origin, picked_data], axis=0)
        train = origin
    train.to_csv(outfile, index=False)


def augument_data(original_data, da_number):
    """
    :param original_data: a list of orginal data paths    type:list
    :param number: the amount of the data after data augmentation process  type:int
    """
    time_start = time.time()
    data_kda_folders = [path+'_eda_'+str(da_number)+'/' for path in original_data]

    for index, folder_p in enumerate(data_kda_folders):
        # print(index,'   ',folder_p)
        subprocess.getstatusoutput('rm -rf ' + folder_p)
        subprocess.getstatusoutput('cp -rf ' + original_data[index] + ' ' + folder_p)

        for root, dirs_p, files in os.walk(folder_p):
            for dir in dirs_p:
                for root, dirs, files in os.walk(folder_p + dir):
                    eda_data(folder_p + dir + '/train.csv', folder_p + dir + '/tr', da_number)
                    break
        subprocess.getstatusoutput('find ' + folder_p + ' -name train.csv |xargs rm')
        subprocess.getstatusoutput('find ' + folder_p +'. -name t | xargs -i mv {} {}rain.csv')

    print("Augmenting all the data, take times :" + str((time.time() - time_start) / 60) + ' mins')


if __name__=="__main__":
    # To augment 5000 data
    augument_data(['data/data_500', 'data/data_2000'], 5000)