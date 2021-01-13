import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import Transformer

class GERL(nn.Module):
    def __init__(self, cfg):
        super(GERL, self).__init__()
        self.cfg = cfg
        self.transformer = Transformer(cfg)
        self.user_emb = nn.Embedding(cfg.user_num, cfg.ID_dim)
        self.news_emb = nn.Embedding(cfg.news_num, cfg.ID_dim)

    def forward(self, data, test_mode=False):
        neg_num = self.cfg.neg_count
        if test_mode:
            neg_num = 0
        # traget and D neighbor user
        all_user = data[:, 0: 1 + self.cfg.D]
        # candidate news neg_num + 1; user his news; neighbor news D
        news_num = neg_num + 1 + self.cfg.max_hist_length + self.cfg.D
        all_news_ID = data[:, 1 + self.cfg.D: 1 + self.cfg + news_num]
        all_news_info = data[:, 1 + self.cfg + news_num:].view(-1, news_num, 1 + self.cfg.max_title_len)
        
        # encode
        all_user = self.user_emb(all_user)
        target_user = all_user[:, 0, :].squeeze(1)
        neighbor_user = all_user[:, 1:, :]
        
        all_news_ID = self.news_emb(all_news_ID)
        candidates_news_ID = all_news_ID[:, :neg_num + 1, :]
        his_news_ID = all_news_ID[:, neg_num + 1: neg_num + 1 + self.cfg.max_hist_length, :]
        neighbor_news_ID = all_news_ID[:, neg_num + 1 + self.cfg.max_hist_length:, :]
        
        all_news_info = self.transformer(all_news_info)
        candidcates_news_info = all_news_info[:, :neg_num + 1, :]
        his_news_info = all_news_info[:, neg_num + 1: neg_num + 1 + self.cfg.max_hist_length, :]
        neighbor_news_info = all_news_info[:, neg_num + 1 + self.cfg.max_hist_length:, :]
        
        