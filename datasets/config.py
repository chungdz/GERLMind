import json
import pickle

class Args():
    def __init__(self):
        self.word_num = 80000
        self.user_num = 70000
        self.news_num = 60000
        self.topic_num = 6000
        
        self.max_hist_length = 50
        self.max_title_len = 30
        self.neg_count = 4
        self.word_dim = 300
        self.topic_dim = 128
        self.ID_dim = 256
        self.hidden_size = 128
        self.head_dim = 16
        self.head_num = 8
        self.dropout = 0.2
        self.D = 15
        
        return None