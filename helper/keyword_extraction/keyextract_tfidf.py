#!/usr/bin/python
# coding=utf-8
# 采用TF-IDF方法提取文本关键词
# http://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting
import sys,codecs
import pandas as pd
import numpy as np
import jieba.posseg
import jieba.analyse
# from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
"""
       TF-IDF权重：
           1、CountVectorizer 构建词频矩阵
           2、TfidfTransformer 构建tfidf权值计算
           3、文本的关键字
           4、对应的tfidf矩阵
"""
# 数据预处理操作：分词，去停用词，词性筛选
def dataPrepos(text, stopkey,pos):
    l = []
    # pos = ['n','v','vn']
    #'nz',#名词
    #'v',
    # 'vd',
    #'vn',#动词
    #'l',
    #'a',#形容词
    # 'd'#副词
    #]  # 定义选取的词性
    seg = jieba.posseg.cut(text)  # 分词
    for i in seg:
        if i.word not in stopkey and i.flag in pos:  # 去停用词 + 词性筛选
        # if i.word not in stopkey:  # 去停用词 + 词性筛选
            l.append(i.word)

    return l


def preprocess_for_corpus(text, stopkey,pos):
    l = []
    seg = jieba.posseg.cut(text)
    for i in seg:
        if i.word not in stopkey and i.flag in pos:  # 去停用词 + 词性筛选
            l.append(i.word)
    return ' '.join(l)

# tf-idf获取文本top10关键词
def getKeywords_tfidf(data,stopkey,topK,pos):
    idList, titleList, abstractList = data['id'], data['title'], data['abstract']
    corpus = [] # 将所有文档输出到一个list中，一行就是一个文档
    for index in range(len(idList)):
        text = '%s。%s' % (titleList[index], abstractList[index]) # 拼接标题和摘要
        text = dataPrepos(text,stopkey,pos) # 文本预处理
        text = " ".join(text) # 连接成字符串，空格分隔
        corpus.append(text)

    # 1、构建词频矩阵，将文本中的词语转换成词频矩阵
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus) # 词频矩阵,a[i][j]:表示j词在第i个文本中的词频
    # 2、统计每个词的tf-idf权值
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    # 3、获取词袋模型中的关键词
    word = vectorizer.get_feature_names()
    # 4、获取tf-idf矩阵，a[i][j]表示j词在i篇文本中的tf-idf权重
    weight = tfidf.toarray()
    # 5、打印词语权重
    ids, titles, keys = [], [], []
    for i in range(len(weight)):
        print(u"-------这里输出第", i+1 , u"篇文本的词语tf-idf------")
        ids.append(idList[i])
        titles.append(titleList[i])
        df_word,df_weight = [],[] # 当前文章的所有词汇列表、词汇对应权重列表
        for j in range(len(word)):
            print( word[j], weight[i][j])
            df_word.append(word[j])
            df_weight.append(weight[i][j])
        df_word = pd.DataFrame(df_word, columns=['word'])
        df_weight = pd.DataFrame(df_weight, columns=['weight'])
        word_weight = pd.concat([df_word, df_weight], axis=1) # 拼接词汇列表和权重列表
        word_weight = word_weight.sort_values(by="weight", ascending = False) # 按照权重值降序排列
        keyword = np.array(word_weight['word']) # 选择词汇列并转成数组格式
        word_split = [keyword[x] for x in range(0,topK)] # 抽取前topK个词汇作为关键词
        word_split = " ".join(word_split)
        # keys.append(word_split.encode("utf-8"))
        keys.append(word_split)

    result = pd.DataFrame({"id": ids, "title": titles, "key": keys},columns=['id','title','key'])
    return result


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results

def getKeywords(filename, data, stopkey, pos):
    labels, abstractList = data['label'], data['content']
    data['ci'] = data.apply(lambda row: preprocess_for_corpus(row['content'], stopkey, pos), axis=1)
    corpus = data['ci'].tolist()

    # 1、构建词频矩阵，将文本中的词语转换成词频矩阵
    # very important, in default case of CV, it will drop word with less than 2 of the length of the word's chars
    vectorizer = CountVectorizer(min_df=1, token_pattern='(?u)\\b\\w+\\b')
    print('getKeywords   2')
    X = vectorizer.fit_transform(corpus) # 词频矩阵,a[i][j]:表示j词在第i个文本中的词频
    print('getKeywords   3')
    # 2、统计每个词的tf-idf权值
    transformer = TfidfTransformer()
    print('getKeywords   4')
    tfidf = transformer.fit_transform(X)
    word = vectorizer.get_feature_names()
    # 3、获取词袋模型中的关键词

    # sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(tfidf.tocoo())

    # extract only the top n; n here is 10
    keywords = extract_topn_from_vector(word, sorted_items, 1000)
    return keywords



def tfidf_getKeywords(path_in, path_out, path_stop, topk, pos = ['n','v','vn','a','vd']):
    df = pd.read_csv(path_in)
    # 停用词表
    stopkey = [w.strip() for w in codecs.open(path_stop, 'r').readlines()]
    # tf-idf关键词抽取
    df['ci'] = df.apply(lambda row: preprocess_for_corpus(row['content'], stopkey, pos), axis=1)
    corpus = df['ci'].tolist()
    cv = CountVectorizer(min_df=1, token_pattern='(?u)\\b\\w+\\b')
    # 统计每个词的tf-idf权值
    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit_transform(cv.fit_transform(corpus))
    tf_idf_vector = tfidf_transformer.transform(cv.transform([' '.join(corpus)]))
    words = cv.get_feature_names()
    # 3、获取词袋模型中的关键词
    # sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    # extract only the top n; n here is 1000
    keywords = extract_topn_from_vector(words, sorted_items, topk)
    key_words = list(keywords.keys())
    df_toSave = pd.DataFrame({'tfidf_keys': key_words})
    df_toSave.to_csv(path_out, index=False)
    return key_words



def main():
    # 读取数据集
    dataFile = 'data/sample_data.csv'
    data = pd.read_csv(dataFile)
    # 停用词表
    stopkey = [w.strip() for w in codecs.open('data/stopWord.txt', 'r').readlines()]
    # tf-idf关键词抽取
    result = getKeywords_tfidf(data,stopkey,10)
    result.to_csv("result/keys_TFIDF.csv",index=False)

if __name__ == '__main__':
    main()
