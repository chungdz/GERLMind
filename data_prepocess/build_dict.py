import os
import json
import pickle
import argparse
import re
import pandas as pd
import numpy as np

def parse_ent_list(x):
    if x.strip() == "":
        return ''

    return ' '.join([k["WikidataId"] for k in json.loads(x)])

punctuation = '!,;:?"\''
def removePunctuation(text):
    text = re.sub(r'[{}]+'.format(punctuation),'',text)
    return text.strip().lower()

data_path = 'data'

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
word_idx = 1
news_idx = 2
for n, title in all_news[['newsid', "title"]].values:
    news_dict[n] = {}
    news_dict[n]['idx'] = news_idx
    news_idx += 1

    tarr = removePunctuation(title).split()
    wid_arr = []
    for t in tarr:
        if t not in word_dict:
            word_dict[t] = word_idx
            word_idx += 1
        wid_arr.append(word_dict[t])
    cur_len = len(wid_arr)
    if cur_len < 10:
        for l in range(10 - cur_len):
            wid_arr.append(0)
    news_dict[n]['title'] = wid_arr[:10]

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
if cur_len < 10:
    for l in range(10 - cur_len):
        wid_arr.append(0)
news_dict['<pad>']['title'] = wid_arr[:10]
## paddning news for history
news_dict['<his>']= {}
news_dict['<his>']['idx'] = 1
news_dict['<his>']['title'] = list(np.zeros(10))

print('all word', len(word_dict))
print('all news', len(news_dict))
json.dump(news_dict, open('data/news.json', 'w', encoding='utf-8'))
json.dump(word_dict, open('data/word.json', 'w', encoding='utf-8'))

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
    user_idx += 1

user_dict['<pad>'] = {}
user_dict['<pad>']['idx'] = 0

print('User num', len(user_dict))
json.dump(user_dict, open('data/user.json', 'w', encoding='utf-8'))
