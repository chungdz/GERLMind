import os
import json
import pickle
import argparse
import re
import pandas as pd
import numpy as np
from data_prepocess.embd import build_word_embeddings, build_news_embeddings

def parse_ent_list(x):
    if x.strip() == "":
        return ''

    return ' '.join([k["WikidataId"] for k in json.loads(x)])

punctuation = '!,;:?"\''
def removePunctuation(text):
    text = re.sub(r'[{}]+'.format(punctuation),'',text)
    return text.strip().lower()

data_path = 'data'
max_title_len = 30

print("Loading news info")
f_train_news = os.path.join(data_path, "train/news.tsv")
f_dev_news = os.path.join(data_path, "valid/news.tsv")
f_test_news = os.path.join(data_path, "test/news.tsv")

print("Loading training news")
all_news = pd.read_csv(f_train_news, sep="\t", encoding="utf-8",
                        names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                        quoting=3)

print("Loading dev news")
dev_news = pd.read_csv(f_dev_news, sep="\t", encoding="utf-8",
                        names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                        quoting=3)
all_news = pd.concat([all_news, dev_news], ignore_index=True)
print("Loading testing news")
test_news = pd.read_csv(f_test_news, sep="\t", encoding="utf-8",
                        names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                        quoting=3)
all_news = pd.concat([all_news, test_news], ignore_index=True)
all_news = all_news.drop_duplicates("newsid")

news_dict = {}
word_dict = {'<pad>': 0}
topic_dict = {'<pad>': 0}
word_idx = 1
news_idx = 2
topic_idx = 1
for n, title, topic in all_news[['newsid', "title", "subcate"]].values:
    news_dict[n] = {}
    news_dict[n]['idx'] = news_idx
    news_dict[n]['clicked'] = set()
    news_dict[n]['neighbor'] = []
    news_idx += 1

    tarr = removePunctuation(title).split()
    wid_arr = []
    for t in tarr:
        if t not in word_dict:
            word_dict[t] = word_idx
            word_idx += 1
        wid_arr.append(word_dict[t])
    cur_len = len(wid_arr)
    if cur_len < max_title_len:
        for l in range(max_title_len - cur_len):
            wid_arr.append(0)
    if topic not in topic_dict:
        topic_dict[topic] = topic_idx
        topic_idx += 1
    news_dict[n]['title'] = [topic_idx] + wid_arr[:max_title_len]

## paddning news for impression
news_dict['<pad>']= {}
news_dict['<pad>']['idx'] = 0
tarr = removePunctuation("This is the title of the padding news").split()
wid_arr = []
for t in tarr:
    if t not in word_dict:
        word_dict[t] = word_idx
        word_idx += 1
    wid_arr.append(word_dict[t])
cur_len = len(wid_arr)
if cur_len < max_title_len:
    for l in range(max_title_len - cur_len):
        wid_arr.append(0)
news_dict['<pad>']['title'] = [0] + wid_arr[:max_title_len]
news_dict['<pad>']['clicked'] = set()
news_dict['<pad>']['neighbor'] = []
## paddning news for history
news_dict['<his>']= {}
news_dict['<his>']['idx'] = 1
news_dict['<his>']['title'] = [0] + list(np.zeros(max_title_len))
news_dict['<his>']['clicked'] = set()
news_dict['<his>']['neighbor'] = []

print('all word', len(word_dict))
print('all news', len(news_dict))
print('all topics', len(topic_dict))

print("Loading behaviors info")
f_train_beh = os.path.join(data_path, "train/behaviors.tsv")
f_dev_beh = os.path.join(data_path, "valid/behaviors.tsv")
f_test_beh = os.path.join(data_path, "test/behaviors.tsv")

print("Loading training beh")
train_beh = pd.read_csv(f_train_beh, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
print("Loading dev beh")
dev_beh = pd.read_csv(f_dev_beh, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
all_beh = pd.concat([train_beh, dev_beh], ignore_index=True)
print("Loading testing beh")
test_beh = pd.read_csv(f_test_beh, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
all_beh = pd.concat([all_beh, test_beh], ignore_index=True)

print("All beh: {}".format(len(all_beh)))

user_ids = pd.unique(all_beh['uid'])
user_dict = {}
user_idx = 1
for u in user_ids:
    user_dict[u] = {}
    user_dict[u]['idx'] = user_idx
    user_dict[u]['clicked'] = []
    user_dict[u]['neighbor'] = []
    user_idx += 1

user_dict['<pad>'] = {}
user_dict['<pad>']['idx'] = 0
user_dict['<pad>']['clicked'] = []
user_dict['<pad>']['neighbor'] = []

print('User num', len(user_dict))

# build graph dict
for uid, hist in train_beh[["uid", "hist"]].values:
    if str(hist) == 'nan':
        his_list = []
    else:
        his_list = str(hist).strip().split()
    
    user_dict[uid]['clicked'] = his_list
    for h in his_list:
        news_dict[h]['clicked'].add(uid)

build_word_embeddings(word_dict, 'data/glove.840B.300d.txt', 'data/emb.npy')
build_news_embeddings(news_dict, 'data/news_info.npy')
pickle.dump(user_dict, open('data/user.pkl', 'wb'))
pickle.dump(news_dict, open('data/news.pkl', 'wb'))
json.dump(word_dict, open('data/word.json', 'w', encoding='utf-8'))
json.dump(topic_dict, open('data/topic.json', 'w', encoding='utf-8'))
