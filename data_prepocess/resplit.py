import numpy as np
import argparse
import json
import math

parser = argparse.ArgumentParser()

parser.add_argument("--fsamples", default="data/raw/train", type=str,
                    help="Path of the training samples file.")
parser.add_argument("--processes", default=4, type=int,
                    help="Processes number")
parser.add_argument('--filenum', type=int, default=10)

cfg = parser.parse_args()

data_list = []

for i in range(cfg.filenum):
    data_list.append(np.load("{}-{}.npy".format(cfg.fsamples, i)))
datanp = np.concatenate(data_list, axis=0)

sub_len = math.ceil(len(datanp) / cfg.processes)

for i in range(cfg.processes):
    s = i * sub_len
    e = (i + 1) * sub_len
    np.save("{}-{}-new.npy".format(cfg.fsamples, i), datanp[s: e])
