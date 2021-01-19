import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()

        self.h1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        
        self.gate_layer = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, seqs, seq_masks=None):

        gates = self.gate_layer(self.h1(seqs)).squeeze(-1)
        if seq_masks is not None:
            gates = gates.masked_fill(seq_masks == 0, -1e9)
        p_attn = F.softmax(gates, dim=-1)
        p_attn = p_attn.unsqueeze(-1)
        h = seqs * p_attn
        output = torch.sum(h, dim=1)
        return output

class SelfAttention(nn.Module):
    def __init__(self, inputdim, head_num, head_dim):
        super(SelfAttention, self).__init__()
        self.head_num = head_num
        self.head_dim = head_dim
        self.output_dim = self.head_num * self.head_dim
        self.query = nn.Linear(inputdim, self.output_dim, bias=False)
        self.key = nn.Linear(inputdim, self.output_dim, bias=False)
        self.value = nn.Linear(inputdim, self.output_dim, bias=False)
    
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        # Q, K, V shape (batch_size, seq_len, head_num * head_dim)

        Q = Q.view(-1, x.size(1), self.head_num, self.head_dim)
        K = K.view(-1, x.size(1), self.head_num, self.head_dim)
        V = V.view(-1, x.size(1), self.head_num, self.head_dim)
        # Q, K, V shape (batch_size, seq_len, head_num, head_dim)

        Q = Q.permute(0, 2, 1, 3)
        # Q shape = (batchsize, head_num, seq_len, head_dim)
        K = K.permute(0, 2, 3, 1)
        #K_seq shape = (batchsize, head_num, head_dim, seq_len)
        V = V.permute(0, 2, 1, 3)
        # V shape = (batchsize, head_num, seq_len, head_dim)

        A = torch.matmul(Q, K) / math.sqrt(self.head_dim)
        # A shape (batchsize, head_num, seq_len_Q, seq_len_K)
        A = F.softmax(A, dim=-1)
        O = torch.matmul(A, V)
        # O shape = (batch_size, head_num, seq_len, head_dim)
        O = O.permute(0, 2, 1, 3)
        # O shape = (batch_size, seq_len, head_num, head_dim)
        O = torch.reshape(O, (-1, O.size(1), self.output_dim))
        # O shape = (batch_size, seq_len, head_num * head_dim)
        return O

class Transformer(nn.Module):
    def __init__(self, cfg):
        super(Transformer, self).__init__()
        self.cfg = cfg
        self.attention = AttentionLayer(cfg.hidden_size)
        self.multi_attention = SelfAttention(cfg.word_dim, cfg.head_num, cfg.head_dim)
        self.topic_emb = nn.Embedding(cfg.topic_num, cfg.topic_dim)
        self.word_emb = nn.Embedding.from_pretrained(torch.FloatTensor(np.load('data/emb.npy')))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, news_info):
        title = news_info[:, :, 1:]
        topics = news_info[:, :, 0]
        seq_len = title.size(1)
        
        title = self.word_emb(title)
        title = title.view(-1, self.cfg.max_title_len, self.cfg.word_dim)
        title = self.multi_attention(title)
        title = self.dropout(title)
        title = self.attention(title)
        title = title.view(-1, seq_len, self.cfg.hidden_size)
        
        topics = self.topic_emb(topics)
        represent = torch.cat([title, topics], dim=-1)
        
        return represent