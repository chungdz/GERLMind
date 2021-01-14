import os
import json
import pickle
import argparse
import re
import pandas as pd
import numpy as np
from data_prepocess.embd import build_word_embeddings

D = 15

user_dict = pickle.load(open('data/user.pkl', 'rb'))
news_dict = pickle.load(open('data/news.pkl', 'wb'))

for n, info in news_dict.items():
    if len(info['clicked']) < 1:
        continue
    neighbor_news_set = set()
    for u in info['clicked']:
        cur_nlist = user_dict[u]['clicked']
        for neighbor_n in cur_nlist:
            neighbor_news_set.add(neighbor_n)
    cur_len = len(neighbor_news_set)
    neighbor_news_list = list(neighbor_news_set)
    if cur_len >= D:
        info['neighbor']
    info['neighbor'] = neighbor_news_set
