import os
import pickle
import torch
from utils.log import logger

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEFAULT_CONFIG = {
    'model_choose': 'bert',

    'lr': 0.0001,
    'epoch': 160,
    'lr_decay': 0.9,
    'weight_decay': None,
    'batch_size': 60,
    'dropout': 0.5,
    'pretrain_vectors': True,
    'static': False,  # ner/model.py 43
    'embedding_dim': 300,
    'hidden_dim': 200,
    'num_layers': 2,
    'pad_index': 1,
    'vector_path': '',
    # 'tag_num': 3,
    # 'vocabulary_size': 0,
    'word_vocab': None,
    'tag_vocab': None,
    'save_path': './ner_save',

    'bert_embedding_dim': 384,
    'bert_config_json': './ner_save/bert_config.json',
    'bert_weight': './ner_save/bert_weight.bin',
    'warmup_proportion': 0.1
}


class Config(object):
    def __init__(self, word_vocab, tag_vocab, vector_path, **kwargs):
        super(Config, self).__init__()
        for name, value in DEFAULT_CONFIG.items():
            setattr(self, name, value)
        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab
        self.tag_num = len(self.tag_vocab) if tag_vocab else 0
        self.vocabulary_size = len(self.word_vocab) if word_vocab else 0
        self.vector_path = vector_path
        for name, value in kwargs.items():
            setattr(self, name, value)

    @staticmethod
    def load(path=DEFAULT_CONFIG['save_path']):
        # config = None
        config_path = os.path.join(path, 'config.pkl')
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        logger.info('loading config from {}'.format(config_path))
        return config

    def save(self, path=None):
        if not hasattr(self, 'save_path'):
            raise AttributeError('config object must init save_path attribute in init method')
        path = path if path else self.save_path
        if not os.path.isdir(path):
            os.mkdir(path)
        config_path = os.path.join(path, 'config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(self, f)
        logger.info('save config to {}'.format(config_path))