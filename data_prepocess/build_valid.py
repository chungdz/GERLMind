import os
import json
import pickle
import argparse
import math
import pandas as pd
import numpy as np
import multiprocessing as mp
import random
from tqdm import tqdm

random.seed(7)

def build_examples(rank, args, df, news_info, user_info, fout):
    news_num = args.neg_num + 1 + args.max_hist_length + args.D * (args.neg_num + 1)
    data_list = []
    for imp_id, hist, imp, user_id in tqdm(df[["id", "hist", "imp", "uid"]].values, total=df.shape[0]):
        if str(hist) == 'nan':
            his_list = []
        else:
            his_list = str(hist).strip().split()

        his_idx_list = [news_info[h]['idx'] for h in his_list]
        
        hislen = len(his_list)
        if hislen < args.max_hist_length:
            for _ in range(args.max_hist_length - hislen):
                his_idx_list.append(news_info['<his>']['idx'])
        else:
            his_idx_list = his_idx_list[-args.max_hist_length:]

        imp_list = str(imp).split(' ')
        
        for impre in imp_list:
            arr = impre.split('-')
            label = int(arr[1])
            
            new_row = []
            new_row.append(int(imp_id))
            new_row.append(label)
            # user idx
            new_row.append(user_info[user_id]['idx'])
            new_row += user_info[user_id]['neighbor']
            # news idx
            new_row.append(news_info[arr[0]]['idx'])
            new_row += his_idx_list
            new_row += news_info[arr[0]]['neighbor']
            assert(len(new_row) == 2 + 1 + args.D + news_num)
            data_list.append(new_row)
    
    datanp = np.array(data_list, dtype=int)
    np.save(fout, datanp)
    print(datanp.shape)

def main(args):
    f_train_beh = os.path.join("data", args.fsamples)
    df = pd.read_csv(f_train_beh, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
    news_info = pickle.load(open('data/news_n.pkl', 'rb'))
    user_info = pickle.load(open('data/user_n.pkl', 'rb'))

    subdf_len = math.ceil(len(df) / args.processes)
    cut_indices = [x * subdf_len for x in range(1, args.processes)]
    dfs = np.split(df, cut_indices)

    processes = []
    for i in range(args.processes):
        output_path = os.path.join("data", args.fout,  "dev-{}.npy".format(i))
        p = mp.Process(target=build_examples, args=(
            i, args, dfs[i], news_info, user_info, output_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path options.
    parser.add_argument("--fsamples", default="valid/behaviors.tsv", type=str,
                        help="Path of the training samples file.")
    parser.add_argument("--fout", default="raw", type=str,
                        help="Path of the output dir.")
    parser.add_argument("--max_hist_length", default=50, type=int,
                        help="Max length of the click history of the user.")
    parser.add_argument("--D", default=15, type=int,
                        help="neighbor num")
    parser.add_argument("--neg_num", default=0, type=int)
    parser.add_argument("--processes", default=40, type=int,
                        help="Processes number")

    args = parser.parse_args()

    main(args)

