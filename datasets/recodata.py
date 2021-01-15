from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import torch
 
class RecoData(Dataset):
    def __init__(self, cfg, bfile, embfile):
        self.file = torch.LongTensor(bfile)
        self.cfg = cfg
        self.embfile = torch.LongTensor(embfile)

    def __getitem__(self, index):
        news_idx = self.file[index, 2 + 1 + self.cfg.D:]
        news_title = self.embfile[news_idx].view(-1)

        return torch.cat([self.file[index], news_title])
 
    def __len__(self):
        return self.file.size(0)

