import json
import pickle

class ModelConfig():
    def __init__(self):

        user_dict = pickle.load(open('data/user_n.pkl', 'rb'))
        news_dict = pickle.load(open('data/news_n.pkl', 'rb'))
        word_dict = json.load(open('data/word.json', 'r', encoding='utf-8'))
        topic_dict = json.load(open('data/topic.json', 'r', encoding='utf-8'))

        self.user_num = len(user_dict)
        self.news_num = len(news_dict)
        self.word_num = len(word_dict)
        self.topic_num = len(topic_dict)

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