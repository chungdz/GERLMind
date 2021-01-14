import os
import json
import pickle
import argparse
import re
import random
import pandas as pd
import numpy as np
from data_prepocess.embd import build_word_embeddings
from tqdm import tqdm

D = 15
with open('data/user.pkl', 'rb') as f:
    user_dict = pickle.load(f)
with open('data/news.pkl', 'rb') as f2:
    news_dict = pickle.load(f2)

for n, info in tqdm(news_dict.items(), total=len(news_dict), desc='news neighbor'):
    if len(info['clicked']) < 1:
        continue
    neighbor_news_set = set()
    for u in info['clicked']:
        cur_nlist = user_dict[u]['clicked']
        for neighbor_n in cur_nlist:
            neighbor_news_set.add(news_dict[neighbor_n]['idx'])
    
    cur_len = len(neighbor_news_set)
    neighbor_news_list = list(neighbor_news_set)
    if cur_len >= D:
        info['neighbor'] = random.sample(neighbor_news_list, D)
    else:
        info['neighbor'] = neighbor_news_list
        for t in range(D - cur_len):
            info['neighbor'].append(news_dict['<his>']['idx'])

for u, info in tqdm(user_dict.items(), total=len(user_dict), desc='user neighbor'):
    if len(info['clicked']) < 1:
        continue
    neighbor_user_dict = {}
    for n in info['clicked']:
        cur_ulist = news_dict[n]['clicked']
        for neighbor_u in cur_ulist:
            cur_idx = user_dict[neighbor_u]['idx']
            if cur_idx not in neighbor_user_dict:
                neighbor_user_dict[cur_idx] = 1
            else:
                neighbor_user_dict[cur_idx] += 1
    
    cur_len = len(neighbor_user_dict)
    if cur_len >= D:
        neighbor_user_list = sorted(neighbor_user_dict, key=lambda x: -neighbor_user_dict[x])[:D]
    else:
        neighbor_user_list = list(neighbor_user_dict)
        for t in range(D - cur_len):
            neighbor_user_list.append(user_dict['<pad>']['idx'])
    info['neighbor'] = neighbor_user_list

with open('data/user_n.pkl', 'wb') as f3:
    user_dict = pickle.dump(user_dict, f3)
with open('data/news_n.pkl', 'wb') as f4:
    news_dict = pickle.dump(news_dict, f4)
