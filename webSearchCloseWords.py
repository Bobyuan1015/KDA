# -*- coding: utf-8 -*-
"""
   File Name：     url_content.py
   Description :  the main detail logic of auditing url sms
   Author :       yuanfang
   date：         2019/12/13
"""

from lxml import html
import pandas as pd
import sys
import os
import pathlib
import re
import requests

project_path = str(pathlib.Path(os.path.abspath(os.curdir)))
sys.path.append(project_path)
print(sys.path)

headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36'}

xpaths=['/html/body/div[1]/div[2]/div[2]/p[11]//text()',
        '/html/body/div[1]/div[2]/div[2]/p[6]//text()',
        '/html/body/div[1]/div[2]/div[2]/p[8]//text()',
        '/html/body/div[1]/div[2]/div[2]/p[13]//text()',
        '/html/body/div[1]/div[2]/div[2]/p[2]//text()']

def web_search(text,closeWords=None):
    """Get the industry info of the sms from baidu search .
    :param text: sms's signature.        type: str
    :return: the content of the sms's company's abstract info    type:str
    """
    if len(get_chinese(text)) <1:
        return '0'
    if closeWords != '0':
        return closeWords
    def getci(text):
        tree = html.fromstring(text)
        close_words = []
        for xpath_ in xpaths:
            text = tree.xpath(xpath_)
            if len(text) > 0:
                for ci in text:
                    close_words.extend(ci.split())
                    print('close:',close_words)
        return list(set(close_words))

    print('web_search ->', text)
    while True:  # 一直循环，知道访问站点成功
        try:
            page = requests.get('https://kmcha.com/similar/' + text, headers=headers, timeout=2)
            # print(page.text)
            close_words = match_ci(page.text)
            # print(close_words)
            # print('  近义词:',content)

            return ','.join(close_words)
            # print('response:',response)
            # response = requests.get(url)
            # content = response.content.decode()
            # print('content:', content)
            # return test_remove_redundants(content)

        except requests.exceptions.ConnectionError:
            print('ConnectionError -- please wait 3 seconds')
            return '0'
            # time.sleep(3)
        except requests.exceptions.ChunkedEncodingError:
            print('ChunkedEncodingError -- please wait 3 seconds')
            # time.sleep(3)
            return '0'
        except Exception as e:
            print('Unfortunitely -- An Unknow Error Happened, Please wait 3 seconds e:', e)
            # time.sleep(3)
            return '0'


def get_chinese(content):
    """
    pick chinese only from a text
    :param text:           type: str
    :return: chines text   type: str

    """
    print('content:',content)
    return re.sub('[^\u4e00-\u9fff]+', '', content)

def remove_redundant(text):
    words = text.split('的同义词')
    return list(set(words))

stops=['的']
def match_ci(text):
    start='的相似词'
    end='热门查询'
    close_words=[]
    if start in text and end in text:
        start_index = text.find(start)+len(start)
        end_index = text.find(end)
        ci_sentences = text[start_index:end_index]
        temp = [close_words.extend(remove_redundant(get_chinese(s.strip()))) for s in ci_sentences.split('&nbsp;')]
        cis = [ci for ci in close_words if len(ci) > 0 and ci not in stops]
        return cis


# df = pd.read_csv('key.csv')
# print(type(df))
# print(df.columns)
# # df.drop(df[df.keys.isnull()].index,inplace=True)
# df['closed_words'] = df['keys'].apply(web_search)
# df.to_csv('done_keys.csv',index=False)