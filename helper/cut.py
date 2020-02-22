# -*- coding: utf-8 -*
from multiprocessing import cpu_count

print(cpu_count())
num_cores =cpu_count()*3
num_partitions =num_cores
from multiprocessing import Pool
from functools import wraps
import time
import subprocess
import os
from helper.keyword_extraction.keyextract_tfidf import *
from helper.keyword_extraction.keyextract_textrank import *
import synonyms
def func_timer(function):
    '''
    timer
    :param function: counting time consumption
    :return: None
    '''
    @wraps(function)
    def function_timer(*args, **kwargs):
        print('[Function: {name} start...]'.format(name = function.__name__))
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print('[Function: {name} finished, spent time: {time:.2f}s]'.format(name = function.__name__,time = t1 - t0))
        return result
    return function_timer


@func_timer
def merge_path(path, out_file_name, sub_name_in_file='clean'):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('max_colwidth', 100)

    # -----------merge all files
    print('merge_path:',path)
    tomerge_files = []
    for root, dirs, files in os.walk(path):

        for file_name in files:
            print('file_name=',file_name)
            # for account_type in accounts:
            if sub_name_in_file in file_name:
                tomerge_files.append(path + file_name)
        break
    print('merge files:',tomerge_files)
    merged_df = merge_csv(tomerge_files, path + out_file_name)
    return merged_df


@func_timer
def merge_csv(files, merge_path, sep=','):

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
            print('open ', f,)
            if sep == '\t':
                df = pd.read_csv(f, lineterminator='\n',delimiter='\t')
            else:
                df = pd.read_csv(f)
            dfs.append(df)
    concat_df = pd.concat(dfs, axis=0, ignore_index=True)
    concat_df.to_csv(merge_path, index=False)
    print('merged_csv.columns= ', concat_df.columns)
    return concat_df


@func_timer
def split_data_by_class(f, dir, length=-1):
    print('split_data_by_class  open:', f, ' save into ',dir)
    # df = pd.read_csv(f, lineterminator='\n', delimiter='\t')
    # df.to_csv(f,index=False)
    # # df.close()
    df = pd.read_csv(f)
    # print(' df.columns=', df.columns)
    classes = df[df.columns[0]].unique()
    # classes = df.columns
    # print('classes=',classes,' df.columns[0]=', df.columns[0])
    files = []
    for cl in classes:
        df1 = df[df[df.columns[0]] == cl]
        file = dir+str(cl)+'.csv'

        df1 = df1.rename(columns={df1.columns[0]: 'label', df1.columns[1]: 'content'})

        if length > 0:
            df1 = df1.iloc[0:length,]
        print(cl,' new len(df1)=', len(df1))
        df1.to_csv(file,index=False)
        files.append(file)
    return files

@func_timer
def get_keywords(f, method='textrank'):
    path_stop = 'stopWord.txt'
    dir_index = f.rfind('/')
    dir = f[:dir_index]
    file_name = f[dir_index + 1:-4]
    topk = 3000
    if file_name == '1' or file_name == '0':
        pos = ['n', 'f', 'nz', 'vd', 'v', 'vd', 'vn', 'a', 'ad', 'an', 'd']
    else:
        pos = ['n', 'f', 'nz', 'vd', 'v', 'vd', 'vn']
    if method == 'textrank':
        path_out = dir + '/' + file_name + 'textrank_key.csv'
        keywords = textrank_getKeywords(f, path_out, path_stop, topk, pos)
    elif method == 'tfidf':
        path_out = dir + '/' + file_name + 'tfidf_key.csv'
        keywords = tfidf_getKeywords(f, path_out, path_stop, topk, pos)
    return keywords

@func_timer
def get_keywords_parallelly(files, func):
    time_start = time.time()
    files_split = np.array_split(files, num_partitions)
    pool = Pool(num_cores)
    pool.map(func, files_split)
    pool.close()
    pool.join()
    print("format the data(remove redundants),            耗时:" + str((time.time() - time_start) / 60) + ' 分')

def compute( df ):
    close_words = []
    print('computing -->len', len(df))
    for index, row in df.iterrows():
        # print(row['keys'])
        # print(type(row.keys))
        # original_word = "人脸"

        original_word = row['keys']
        print(index,' original_word=', original_word)
        word, score = synonyms.nearby(original_word)
        # print('word:', word)
        # print('word:', score)
        selected = []
        for i, s in enumerate(score):
            if s > 0.5 and original_word != word[i]:
                selected.append(word[i])

        close_word = ','.join(selected)
        if len(close_word) == 0:
            close_word = '无'
        print('close_word:', close_word)
        close_words.append(close_word)

    print('computing --done')
    # print('final close_word=type = ', type(close_words))
    df['close_words'] = pd.DataFrame({'close_words': close_words})
    return df


@func_timer
def get_synonyms_parallelly(df, func):
    time_start = time.time()
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    dfs= pool.map(func, df_split)
    print('pool.map len(dfs)=', len(dfs))
    # for i, df1 in enumerate(dfs):
    #     print('save ',i,' file')
    #     df1.to_csv(str(i)+".csv", index=False)

    df = pd.concat(dfs)
    print('concat len(df)=', len(df))
    pool.close()
    pool.join()
    print("get_synonyms_parallelly            耗时:" + str((time.time() - time_start) / 60) + ' 分')
    return df

@func_timer
def rename_columns(files, column_name):
    for file in files:
        print('rename_columns  read=',file)
        df = pd.read_csv(file)
        # print('df.columns=',df.columns)
        df = df.rename(columns={df.columns[2]: column_name})
        df.to_csv(file, index=False)


@func_timer
def compute_keywords():

    # files = [
    #     'data_orginal/chnsenticorp/1.csv',
    #     'data_orginal/chnsenticorp/0.csv',
    #     'data_orginal/weibo_senti_100k/1.csv',
    #     'data_orginal/weibo_senti_100k/0.csv',
    #     'data_orginal/cnews_10/体育.csv',
    #     'data_orginal/cnews_10/娱乐.csv',
    #     'data_orginal/cnews_10/家居.csv',
    #     'data_orginal/cnews_10/房产.csv',
    #     'data_orginal/cnews_10/教育.csv',
    #     'data_orginal/cnews_10/时尚.csv',
    #     'data_orginal/cnews_10/时政.csv',
    #     'data_orginal/cnews_10/游戏.csv',
    #     'data_orginal/cnews_10/科技.csv',
    #     'data_orginal/cnews_10/财经.csv'
        # 'data-500/chnsenticorp/1.csv',
        # 'data-500/chnsenticorp/0.csv',
        # 'data-500/weibo_senti_100k/1.csv',
        # 'data-500/weibo_senti_100k/0.csv',
        #
        # 'data-500/cnews_10/体育.csv',
        # 'data-500/cnews_10/娱乐.csv',
        # 'data-500/cnews_10/家居.csv',
        # 'data-500/cnews_10/房产.csv',
        # 'data-500/cnews_10/教育.csv',
        # 'data-500/cnews_10/时尚.csv',
        # 'data-500/cnews_10/时政.csv',
        # 'data-500/cnews_10/游戏.csv',
        # 'data-500/cnews_10/科技.csv',
        # 'data-500/cnews_10/财经.csv',
        #
        # 'data-2000/cnews_10/体育.csv',
        # 'data-2000/cnews_10/娱乐.csv',
        # 'data-2000/cnews_10/家居.csv',
        # 'data-2000/cnews_10/房产.csv',
        # 'data-2000/cnews_10/教育.csv',
        # 'data-2000/cnews_10/时尚.csv',
        # 'data-2000/cnews_10/时政.csv',
        # 'data-2000/cnews_10/游戏.csv',
        # 'data-2000/cnews_10/科技.csv',
        # 'data-2000/cnews_10/财经.csv',
        # 'data-2000/weibo_senti_100k/1.csv',
        # 'data-2000/weibo_senti_100k/0.csv',
        # 'data-2000/chnsenticorp/1.csv',
        # 'data-2000/chnsenticorp/0.csv'
    # ]
    all_files = ['1.csv', '0.csv', '体育.csv',
                 '娱乐.csv', '家居.csv', '房产.csv',
                 '教育.csv', '时尚.csv', '游戏.csv',
                 '科技.csv', '财经.csv', '时政.csv']
    root_dir = 'data/data_orginal/'
    #1.分别得到每个类别的关键词 == tfidf 交集 textrank
    for root, dirs_p, files in os.walk(root_dir):
        all_keys = []
        final_all_keys = set()
        for dir in dirs_p:
            if dir in ['cnews_10','weibo_senti_100k','chnsenticorp']:
                for root_, dirs_, files_ in os.walk(root_dir+dir):
                    keys_classes = []
                    for file_name in files_:
                        if file_name in all_files:
                            file = root_dir + dir + '/' + file_name
                            print(file)
                            textrank_keys = get_keywords(file, method='textrank')
                            tfidf_keys = get_keywords(file, method='tfidf')
                            jiao = list(set(textrank_keys) & set(tfidf_keys))
                            dir_index = file.rfind('/')
                            save_dir = file[:dir_index]
                            file_name = file[dir_index + 1:-4]
                            key_save_path = save_dir + '/' + file_name + 'size' + str(len(jiao)) + '_inserction_key.csv'
                            key_df = pd.DataFrame({'key': jiao})
                            keys_classes.append(jiao)
                            key_df.to_csv(key_save_path, index=False)
                    keys_a_set = set()
                    for key_a_class in keys_classes:
                        keys_a_set ^= set(key_a_class)
                    all_keys.append(list(keys_a_set))
                    finale_keys_df = pd.DataFrame({'finale_keys': list(keys_a_set)})
                    finale_keys_df.to_csv(root_dir + dir + '/'+dir+'_Size' + str(len(finale_keys_df)) + 'finale_keys.csv', index=False)
        for key_a_set in all_keys:
            final_all_keys ^= set(key_a_set)
        all_keys_df = pd.DataFrame({'finale_all_keys': list(final_all_keys)})
        all_keys_df.to_csv(str(len(all_keys_df)) + 'len_all_keys' + '.csv', index=False)
        return str(len(all_keys_df)) + 'len_all_keys' + '.csv'



    #                 if '_key.csv' in file_name:
    #                     parent_dir = path_class + dir + '/'
    #                     print(parent_dir + file_name)
    #                     pd_file = pd.read_csv(parent_dir + file_name)
    #                     pd_file.drop(pd_file[pd_file.union.isnull()].index, inplace=True)
    #                     all_words=[]
    #                     for words in pd_file['Intersection']:
    #                         all_words.extend(words.split(','))
    #                     for word in set(all_words):
    #                         if len(word) < 1:
    #                             continue
    #                         if word in all_key_words:
    #                             word_index = all_key_words.index(word)
    #                             key_closewords.append(all_key_closewords[word_index])
    #                             key_words.append(word)
    #
    #                     df_new = pd.DataFrame({'key_words':key_words,'close_words':key_closewords})
    #                     df_new.to_csv(parent_dir+str(len(df_new))+'closeWords'+'_'+file_name.split('_key')[0]+'.csv',index=False)
    # for root, dirs, files in os.walk(root_dir):
    #     for _root, _dir, _files in os.walk(dirs):
    #         for file_name in _files:
    #             print(file_name)
                # if '_key.csv' in file_name:
                #     parent_dir = root_dir + dir + '/'
                #     print(parent_dir + file_name)
                #     pd_file = pd.read_csv(parent_dir + file_name)
                #     pd_file.drop(pd_file[pd_file.union.isnull()].index, inplace=True)
                #     all_words = []
                #     for words in pd_file['Intersection']:
                #         all_words.extend(words.split(','))
                #     for word in set(all_words):
                #         if len(word) < 1:
                #             continue
                #         if word in all_key_words:
                #             word_index = all_key_words.index(word)
                #             key_closewords.append(all_key_closewords[word_index])
                #             key_words.append(word)
    # for file in files:
    #     textrank_keys = get_keywords(file, method='textrank')
    #     tfidf_keys = get_keywords(file, method='tfidf')
    #     jiao = list(set(textrank_keys) & set(tfidf_keys))
    #     dir_index = file.rfind('/')
    #     save_dir = file[:dir_index]
    #     file_name = file[dir_index + 1:-4]
    #     key_save_path = save_dir + '/' + file_name + 'size' + str(len(jiao))+'_inserction_key.csv'
    #     key_df = pd.DataFrame({'key': jiao})
    #     key_df.to_csv(key_save_path, index=False)


    # to_merge_files = ['data_orginal/chnsenticorp/1.csv', 'data_orginal/chnsenticorp/0.csv',
    #                   'data_orginal/weibo_senti_100k/1.csv', 'data_orginal/weibo_senti_100k/0.csv']
    # merged_df = merge_path(to_merge_files, 'data_orginal/1_0.csv')
    # merged_df.drop_duplicates(subset='sms_content', keep='first', inplace=True)
    # get_keywords_parallelly(files, compute_keywords_textrank)

    # compute_keywords_textrank(files)
    # files = [
    #     'data-500/chnsenticorp/1_key.csv',
    #     'data-500/chnsenticorp/0_key.csv',
    #     'data-500/weibo_senti_100k/1_key.csv',
    #     'data-500/weibo_senti_100k/0_key.csv',
    #
    #     'data-500/cnews_10/体育_key.csv',
    #     'data-500/cnews_10/娱乐_key.csv',
    #     'data-500/cnews_10/家居_key.csv',
    #     'data-500/cnews_10/房产_key.csv',
    #     'data-500/cnews_10/教育_key.csv',
    #     'data-500/cnews_10/时尚_key.csv',
    #     'data-500/cnews_10/时政_key.csv',
    #     'data-500/cnews_10/游戏_key.csv',
    #     'data-500/cnews_10/科技_key.csv',
    #     'data-500/cnews_10/财经_key.csv',
    #
    #     'data-2000/cnews_10/体育_key.csv',
    #     'data-2000/cnews_10/娱乐_key.csv',
    #     'data-2000/cnews_10/家居_key.csv',
    #     'data-2000/cnews_10/房产_key.csv',
    #     'data-2000/cnews_10/教育_key.csv',
    #     'data-2000/cnews_10/时尚_key.csv',
    #     'data-2000/cnews_10/时政_key.csv',
    #     'data-2000/cnews_10/游戏_key.csv',
    #     'data-2000/cnews_10/科技_key.csv',
    #     'data-2000/cnews_10/财经_key.csv',
    #
    #     'data-2000/weibo_senti_100k/1_key.csv',
    #     'data-2000/weibo_senti_100k/0_key.csv',
    #     'data-2000/chnsenticorp/1_key.csv',
    #     'data-2000/chnsenticorp/0_key.csv'
    # ]
    # get_keywords_parallelly(files, compute_keywords_tfidf)
    # compute_keywords_tfidf(files)

@func_timer
def combine_keywords():
    chn_files = [
        'data-500/chnsenticorp/1_key.csv',
        'data-500/chnsenticorp/0_key.csv',
        'data-2000/chnsenticorp/1_key.csv',
        'data-2000/chnsenticorp/0_key.csv'
    ]
    wei_files = [
        'data-500/weibo_senti_100k/1_key.csv',
        'data-500/weibo_senti_100k/0_key.csv',
        'data-2000/weibo_senti_100k/1_key.csv',
        'data-2000/weibo_senti_100k/0_key.csv'
    ]
    cnews_files = [
        'data-500/cnews_10/体育_key.csv',
        'data-500/cnews_10/娱乐_key.csv',
        'data-500/cnews_10/家居_key.csv',
        'data-500/cnews_10/房产_key.csv',
        'data-500/cnews_10/教育_key.csv',
        'data-500/cnews_10/时尚_key.csv',
        'data-500/cnews_10/时政_key.csv',
        'data-500/cnews_10/游戏_key.csv',
        'data-500/cnews_10/科技_key.csv',
        'data-500/cnews_10/财经_key.csv',

        'data-2000/cnews_10/体育_key.csv',
        'data-2000/cnews_10/娱乐_key.csv',
        'data-2000/cnews_10/家居_key.csv',
        'data-2000/cnews_10/房产_key.csv',
        'data-2000/cnews_10/教育_key.csv',
        'data-2000/cnews_10/时尚_key.csv',
        'data-2000/cnews_10/时政_key.csv',
        'data-2000/cnews_10/游戏_key.csv',
        'data-2000/cnews_10/科技_key.csv',
        'data-2000/cnews_10/财经_key.csv',
    ]
    files_set = [chn_files, wei_files, cnews_files]

    outfile_keys = ['key_chn.csv', 'key_wei.csv', 'key_cnews.csv']

    all_keywords = []
    for i, files in enumerate(files_set):
        keywords_of_a_class = []
        for file in files:
            print('open ', file)
            df = pd.read_csv(file)
            jiaoji = []
            binji = []

            for index, row in df.iterrows():
                # print(file, '  ', index, '             ', row)
                if isinstance(row.textrank_keys, float):
                    aa = ['']
                else:
                    aa = row.textrank_keys.split(',')
                if isinstance(row.textrank_keys, float):
                    bb = ['']
                else:
                    bb = row.textrank_keys.split(',')

                keywords_of_a_class.extend(aa)
                keywords_of_a_class.extend(bb)
                all_keywords.extend(aa)
                all_keywords.extend(bb)

                bin = list(set(aa) | set(bb))
                bin = ','.join(bin)
                jiao = list(set(aa) & set(bb))
                jiao = ','.join(jiao)
                binji.append(bin)
                jiaoji.append(jiao)

            df['Intersection'] = pd.DataFrame({'Intersection': jiaoji})
            df['union'] = pd.DataFrame({'union': binji})
            df.to_csv(file, index=False)
        print('len(keywords_of_a_class)=', len(keywords_of_a_class), '  len(list(set(keywords_of_a_class)))=', len(list(set(keywords_of_a_class))),' len(all_keywords)=',len(all_keywords))
        keywords_a_class = list(set(keywords_of_a_class))
        df_a_class = pd.DataFrame({"keys": keywords_a_class})
        df_a_class.to_csv(outfile_keys[i],index=False)
    print(' len(all_keywords)=',len(all_keywords), ' len(list(set(all_keywords)))=', len(list(set(all_keywords))))
    df_all = pd.DataFrame({"keys": list(set(all_keywords))})
    df_all.to_csv('key.csv', index=False)


def get_keys_of_class(path_source, path_class):
    df = pd.read_csv(path_source)
    df.drop(df[df.key_words.isnull()].index,inplace=True)
    df.drop(df[df.close_words.isnull()].index, inplace=True)
    all_key_words = df['key_words'].tolist()

    all_key_closewords = df['close_words'].tolist()
    # for index, words in enumerate(df['close_words']):
    #     # print('close_words index=', index,' ',words)
    #     all_key_closewords.extend(words.split(','))


    for root, dirs_p, files in os.walk(path_class):
        for dir in dirs_p:
            key_words=[]
            key_closewords=[]
            for root, dirs, files in os.walk(path_class+dir):
                for file_name in files:
                    if '_key.csv' in file_name:
                        parent_dir = path_class + dir + '/'
                        print(parent_dir + file_name)
                        pd_file = pd.read_csv(parent_dir + file_name)
                        pd_file.drop(pd_file[pd_file.union.isnull()].index, inplace=True)
                        all_words=[]
                        for words in pd_file['Intersection']:
                            all_words.extend(words.split(','))
                        for word in set(all_words):
                            if len(word) < 1:
                                continue
                            if word in all_key_words:
                                word_index = all_key_words.index(word)
                                key_closewords.append(all_key_closewords[word_index])
                                key_words.append(word)

                        df_new = pd.DataFrame({'key_words':key_words,'close_words':key_closewords})
                        df_new.to_csv(parent_dir+str(len(df_new))+'closeWords'+'_'+file_name.split('_key')[0]+'.csv',index=False)


def merge_keys_by_class(dirs,class_flags, outfile):
    for class_flag in class_flags:
        merge_paths = []
        for p_dir in dirs:
            for root, _dirs, files in os.walk( p_dir):
                for dir in _dirs:
                    for root, _dirs, files in os.walk(p_dir+dir):
                        for file_name in files:
                            if class_flag in file_name:
                                path_file = p_dir+dir + '/' + file_name
                                merge_paths.append(path_file)
                                print(path_file)

        merged_path = merge_csv(merge_paths, outfile+class_flag+'.csv')
        df = pd.read_csv(merged_path)
        df.drop_duplicates(subset=['key_words', 'close_words'], keep='first', inplace=True)  # remove duplication rows
        df.to_csv(outfile+str(len(df))+class_flag+'.csv',index=False)


def replace_words(file_in, file_out):
    df_in = pd.read_csv(file_in)
    df_out = pd.read_csv(file_out)
    df_out['close_words'] = '0'
    for i_row, row in df_in.iterrows():
        df_out.ix[df_out['finale_all_keys'] == row['key_words'], 'close_words'] = row['close_words']
        # df_out.loc[df_out['finale_all_keys'] == row['key_words']] = row['close_words']
    df_out.fillna('0')
    df_out.to_csv(file_out,index=False)



def generate_small_data_set():

    subprocess.getstatusoutput('rm -rf data/data_orginal/cnews_10/whole.csv')
    subprocess.getstatusoutput('rm -rf data/data_orginal/chnsenticorp/whole.csv')
    subprocess.getstatusoutput('rm -rf data/data_orginal/weibo_senti_100k/whole.csv')
    subprocess.getstatusoutput('rm -rf data/data_500/cnews_10/*')
    subprocess.getstatusoutput('rm -rf data/data_500/chnsenticorp/*')
    subprocess.getstatusoutput('rm -rf data/data_500/weibo_senti_100k/*')
    subprocess.getstatusoutput('rm -rf data/data_2000/cnews_10/*')
    subprocess.getstatusoutput('rm -rf data/data_2000/chnsenticorp/*')
    subprocess.getstatusoutput('rm -rf data/data_2000/weibo_senti_100k/*')
    sizes = [500, 2000]
    datasets = ['chnsenticorp', 'cnews_10', 'weibo_senti_100k']
    for dataset in datasets:
        dir = 'data/data_orginal/' + dataset + '/'
        original_files = [dir + 'train.csv', dir + 'test.csv', dir + 'dev.csv']
        merge_csv(original_files, dir + 'whole.csv')
        split_data_by_class(dir + 'whole.csv', dir)
        for size in sizes:
            dir_sub_data_set = 'data/data_' + str(size) + '/' + dataset + '/'
            subprocess.getstatusoutput('mkdir -p ' + dir_sub_data_set)
            split_data_by_class(dir + 'whole.csv', dir_sub_data_set, size)
            merge_path(dir_sub_data_set, 'train.csv', '')
            print('cp -rf ' + dir + '/dev.csv ' + dir + 'test.csv  ' + dir_sub_data_set)
            subprocess.getstatusoutput('cp -rf ' + dir + 'dev.csv ' + dir + 'test.csv  ' + dir_sub_data_set)





if __name__ == '__main__':

    #1.generate 500 2000 size data set 生成小样本数据集
    generate_small_data_set()
    #2. compute keywords for each class
    # compute_keywords()
            # # save_keywords_of_a_class()
            # get_keys_of_class('/home/y/disk-2t/workplace/play/for_paper/kda/findKeys0108.csv',
            #                   '/home/y/disk-2t/workplace/play/for_paper/kda/data-500/')
            # get_keys_of_class('/home/y/disk-2t/workplace/play/for_paper/kda/findKeys0108.csv',
            #                   '/home/y/disk-2t/workplace/play/for_paper/kda/data-2000/')
            #
            # merge_keys_by_class(['/home/y/disk-2t/workplace/play/for_paper/kda/data-500/',
            #                      '/home/y/disk-2t/workplace/play/for_paper/kda/data-2000/'],
            #                     ['closeWords_1','closeWords_0'],
            #                     '/home/y/disk-2t/workplace/play/for_paper/kda/'
            #                     )
            #
            # merge_keys_by_class(['/home/y/disk-2t/workplace/play/for_paper/kda/data-500/',
            #                      '/home/y/disk-2t/workplace/play/for_paper/kda/data-2000/'],
            #                     ['closeWords_房产', 'closeWords_科技','closeWords_游戏','closeWords_财经','closeWords_家居',
            #                      'closeWords_时政', 'closeWords_娱乐','closeWords_时尚','closeWords_体育','closeWords_教育'],
            #                     '/home/y/disk-2t/workplace/play/for_paper/kda/'
            #                     )
    #3. get few found close-words
    # replace_words('/home/y/disk-2t/workplace/play/for_paper/kda/118closewords.csv','/home/y/disk-2t/workplace/play/for_paper/kda/9001len_all_keys.csv')

    #4. get closewords from the website
    # df = pd.read_csv('/home/y/disk-2t/workplace/play/for_paper/kda/9001len_all_keys.csv')
    # # df.fillna('0')
    # df['close_words'] = df.apply(lambda row: web_search(row['finale_all_keys'], row['close_words']), axis=1)
    # df.to_csv('/home/y/disk-2t/workplace/play/for_paper/kda/keys_.csv',index=False)



    # subprocess.getstatusoutput('rm -rf data_orginal/cnews_10/whole.csv')
    # subprocess.getstatusoutput('rm -rf data_orginal/chnsenticorp/whole.csv')
    # subprocess.getstatusoutput('rm -rf data_orginal/weibo_senti_100k/whole.csv')
    # subprocess.getstatusoutput('rm -rf data-500/cnews_10/*')
    # subprocess.getstatusoutput('rm -rf data-500/chnsenticorp/*')
    # subprocess.getstatusoutput('rm -rf data-500/weibo_senti_100k/*')
    # subprocess.getstatusoutput('rm -rf data-2000/cnews_10/*')
    # subprocess.getstatusoutput('rm -rf data-2000/chnsenticorp/*')
    # subprocess.getstatusoutput('rm -rf data-2000/weibo_senti_100k/*')
    # sizes = [500, 2000]
    # datasets = ['chnsenticorp','cnews_10','weibo_senti_100k']
    # # datasets = ['chnsenticorp']
    # for dataset in datasets:
    #     dir='data_orginal/'+dataset+'/'
    #     original_files = [dir+'train.csv', dir+'test.csv', dir+'dev.csv']
    #
    #     cnews_500_whole_file = merge_csv(original_files, dir+'whole.csv')
    #     for size in sizes:
    #         dir_sub_data_set = 'data-'+str(size)+'/'+dataset+'/'
    #         split_data_by_class(dir+'whole.csv',dir_sub_data_set, size)
    #         merge_path(dir_sub_data_set, 'train.csv','')
    #         print('cp -rf '+dir+'/dev.csv '+dir+'test.csv  '+dir_sub_data_set)
    #         subprocess.getstatusoutput('cp -rf '+dir+'dev.csv '+dir+'test.csv  '+dir_sub_data_set)
    #
    # subprocess.getstatusoutput('find . -name  *_key.csv | xargs rm ')
    # subprocess.getstatusoutput('rm ')
    # compute_keywords_tfidf(['data-500/cnews_10/体育_key.csv'])
    # compute_keywords()
    # combine_keywords()
    # save_keywords_of_a_class()
    # cnews_2000_whole_file = merge_csv(cnews_files, 'data-2000/cnews_10/whole.csv')
    # split_data_by_class('data-2000/cnews_10/whole.csv', 'data-2000/cnews_10/', 2000)

    # cnews_2000_files = ['data-2000/cnews_10/train.csv', 'data-2000/cnews_10/test.csv', 'data-2000/cnews_10/dev.csv']
    # cnews_2000_whole_file = merge_csv(cnews_2000_files, 'data-2000/cnews_10/whole.csv')

    # chnsenticorpfiles=['data/chnsenticorp/train.csv','data/chnsenticorp/test.csv','data/chnsenticorp/test.csv']
    # chnsenticorp_path = merge_csv(chnsenticorpfiles,'data/chnsenticorp/whole.csv')
    # split_data_by_class('data/chnsenticorp/whole.csv','data/chnsenticorp/')
    # split_data_by_class('data/weibo_senti_100k/weibo_senti_100k.csv', "data/weibo_senti_100k/")

    # thuCNewsfiles=['data/cnews_10/train.csv','data/cnews_10/test.csv','data/cnews_10/dev.csv']
    # thuCNews_path = merge_csv(thuCNewsfiles,'data/cnews_10/whole.csv')
    # split_data_by_class('data/cnews_10/whole.csv', "data/cnews_10/")
    # files = [
    #     # 'data/chnsenticorp/1.csv',
    #     # 'data/chnsenticorp/0.csv',
    #     #'data/weibo_senti_100k/1.csv',
    #     #'data/weibo_senti_100k/0.csv',
    #     # 'data/cnews_10/股票.csv',
    #     # 'data/cnews_10/星座.csv',
    #     'data/cnews_10/体育.csv',
    #     'data/cnews_10/娱乐.csv',
    #     'data/cnews_10/家居.csv',
    #     # 'data/cnews_10/彩票.csv',
    #     'data/cnews_10/房产.csv',
    #     'data/cnews_10/教育.csv',
    #     'data/cnews_10/时尚.csv',
    #     'data/cnews_10/时政.csv',
    #     'data/cnews_10/游戏.csv',
    #     # 'data/cnews_10/社会.csv',
    #     'data/cnews_10/科技.csv',
    #     'data/cnews_10/财经.csv'
    # ]
    # compute(df)
    # get_keywords_parallelly(files, compute_keywords_textrank)


    # get_keywords_parallelly(files, compute_keywords_tfidf)

    # split_data_by_class('data/weibo_senti_100k/weibo_senti_100k.csv','data/weibo_senti_100k/')

    # for root, dirs, files in os.walk('data/THUCNews'):
    #     print('dirs:',dirs)
    #     break


