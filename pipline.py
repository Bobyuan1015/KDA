import os

from helper.data_preprocess import preprocess_chnsenticorp, preprocess_cnews, preprocess_weibo
from helper.cut import generate_small_data_set, compute_keywords
from helper.webSearchCloseWords import web_search_close_keys

#1. preprocess all dataset
preprocess_weibo()
preprocess_chnsenticorp()
preprocess_cnews()

#2.cut small dataset
generate_small_data_set()

# 3. compute keywords for each classx
keywords_file = compute_keywords()

#4. get closewords from the website
web_search_close_keys(keywords_file)

#5. keywords data augmentation
os.system('python kda.py')

#6. train all dataset
os.system('./start.sh')
