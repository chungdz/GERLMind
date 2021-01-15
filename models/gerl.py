import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import Transformer, AttentionLayer

class GERL(nn.Module):
    def __init__(self, cfg):
        super(GERL, self).__init__()
        self.cfg = cfg
        self.transformer = Transformer(cfg)
        self.user_emb = nn.Embedding(cfg.user_num, cfg.ID_dim)
        self.news_emb = nn.Embedding(cfg.news_num, cfg.ID_dim)
        self.his_news_att = AttentionLayer(cfg.hidden_size * 2)
        self.neighbor_user_att = AttentionLayer(cfg.ID_dim)
        self.neighbor_news_ID_att = AttentionLayer(cfg.ID_dim)
        self.neighbor_news_info_att = AttentionLayer(cfg.hidden_size * 2)

    def forward(self, data, test_mode=False):
        neg_num = self.cfg.neg_count
        if test_mode:
            neg_num = 0
        # traget and D neighbor user
        all_user = data[:, 0: 1 + self.cfg.D]
        # candidate news neg_num + 1; user his news; neighbor news D
        news_num = neg_num + 1 + self.cfg.max_hist_length + self.cfg.D * (neg_num + 1)
        all_news_ID = data[:, 1 + self.cfg.D: 1 + self.cfg.D + news_num]
        all_news_info = data[:, 1 + self.cfg.D + news_num:].view(-1, news_num, 1 + self.cfg.max_title_len)
        
        # encode
        all_user = self.user_emb(all_user)
        target_user = all_user[:, 0, :].squeeze(1)
        neighbor_user = all_user[:, 1:, :]
        
        all_news_ID = self.news_emb(all_news_ID)
        candidates_news_ID = all_news_ID[:, :neg_num + 1, :]
        his_news_ID = all_news_ID[:, neg_num + 1: neg_num + 1 + self.cfg.max_hist_length, :]
        neighbor_news_ID = all_news_ID[:, neg_num + 1 + self.cfg.max_hist_length:, :].reshape(-1, self.cfg.D, self.cfg.ID_dim)
        
        all_news_info = self.transformer(all_news_info)
        candidcates_news_info = all_news_info[:, :neg_num + 1, :]
        his_news_info = all_news_info[:, neg_num + 1: neg_num + 1 + self.cfg.max_hist_length, :]
        neighbor_news_info = all_news_info[:, neg_num + 1 + self.cfg.max_hist_length:, :].reshape(-1, self.cfg.D, self.cfg.hidden_size * 2)

        # calculate six represents
        nt_one = candidcates_news_info
        ne_two = self.neighbor_news_ID_att(neighbor_news_ID).reshape(-1, neg_num + 1, self.cfg.ID_dim)
        nt_two = self.neighbor_news_info_att(neighbor_news_info).reshape(-1, neg_num + 1, self.cfg.hidden_size * 2)
        # print(nt_one.size(), ne_two.size(), nt_two.size())
        ut_one = self.his_news_att(his_news_info)
        ue_one = target_user
        ue_two = self.neighbor_user_att(neighbor_user)
        # print(ut_one.size(), ue_one.size(), ue_two.size())

        news_represent = torch.cat([nt_two, ne_two, nt_one], dim=-1)
        user_represent = torch.cat([ue_one, ue_two, ut_one], dim=-1)

        user_represent = user_represent.repeat(1, neg_num + 1).view(-1, neg_num + 1, news_represent.size(-1))
        similarity = torch.sum(user_represent * news_represent, dim=-1)

        return similarity

        