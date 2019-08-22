import os
import pickle
import torch
import torch.nn as nn
from crf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.vocab import Vectors
from config import DEVICE
from utils.log import logger


class BaseModel(nn.Module):  # 定义了模型的加载和保存的函数
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args
        self.save_path = args.save_path

    def load(self, path=None):
        path = path if path else self.save_path
        map_location = None if torch.cuda.is_available() else 'cpu'
        model_path = os.path.join(path, 'model.pkl')
        self.load_state_dict(torch.load(model_path, map_location=map_location))
        logger.info('loading model from {}'.format(model_path))

    def save(self, path=None):
        path = path if path else self.save_path
        if not os.path.isdir(path):
            os.mkdir(path)
        model_path = os.path.join(path, 'model.pkl')
        torch.save(self.state_dict(), model_path)
        logger.info('saved model to {}'.format(model_path))


class BiLstmCrf(BaseModel):  # 模型网络定义，BiLstm共4层：embedding lstm linear crf
    def __init__(self, args):
        super(BiLstmCrf, self).__init__(args)

        self.args = args
        self.vector_path = args.vector_path  # 预训练词向量的路径 txt
        self.hidden_dim = args.hidden_dim  # 隐藏层
        self.tag_num = args.tag_num
        self.batch_size = args.batch_size
        self.bidirectional = True          # BiLstm
        self.num_layers = args.num_layers
        self.pad_index = args.pad_index
        self.dropout = args.dropout      # 训练时网络中连接 dropout 的概率
        self.save_path = args.save_path

        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim

        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension).to(DEVICE)
        self.drop = nn.Dropout(0.8)
        if args.pretrain_vectors and args.word_vocab:  # 加载与训练词向量到 embedding 层
            weight = torch.randn(len(args.word_vocab), args.embedding_dim)

            with open(args.vector_path, 'rb') as f:
                logger.info('loading word vectors from {}'.format(args.vector_path))
                V = pickle.load(f)
            # V = Vectors(args.vector_path, cache='data')

            for word in args.word_vocab.itos:
                if word in V.stoi:
                    weight[args.word_vocab.stoi[word]] = V.vectors[V.stoi[word]]
            self.embedding = self.embedding.from_pretrained(weight, freeze=args.static).to(DEVICE)

        self.lstm = nn.LSTM(embedding_dimension, self.hidden_dim, bidirectional=self.bidirectional,
                            num_layers=self.num_layers, dropout=self.dropout).to(DEVICE)
        # hidden 除以 2 是因为，双向lstm输出的时候会翻倍，不除以二改成下面的 linear层中 hidden*2 也行
        self.linear1 = nn.Linear(self.hidden_dim*2, self.hidden_dim).to(DEVICE)
        self.lin_drop = nn.Dropout(0.5)
        self.linear2 = nn.Linear(self.hidden_dim, self.tag_num+2).to(DEVICE)  # 隐藏层到 label 的线性转换，即维度变换。
        self.crf_layer = CRF(self.tag_num).to(DEVICE)

    # def init_weight(self):  # 对各层的权重矩阵进行初始化
    #     nn.init.xavier_normal_(self.embedding.weight)
    #
    #     for name, param in self.lstm.named_parameters():
    #         if 'weight' in name:
    #             nn.init.xavier_normal_(param)
    #
    #     nn.init.xavier_normal_(self.linear.weight)

    def init_hidden(self, batch_size=None):  # 生成 lstm 层的输入
        if batch_size is None:
            batch_size = self.batch_size
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(DEVICE)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(DEVICE)
        return h0, c0

    def loss(self, x, sent_lengths, y):
        mask = torch.ne(x, self.pad_index)  # 判断 x 中数字为 1 时(即该位置为pad补丁) mask取0
        emissions = self.lstm_forward(x, sent_lengths)
        emissions = torch.transpose(emissions, 1, 0)  # 矩阵转置为：[batch_size*sentence_length*tag_size]
        mask = torch.transpose(mask, 1, 0)
        y = torch.transpose(y, 1, 0)
        loss_function = self.crf_layer.neg_log_likelihood_loss
        return loss_function(emissions, mask, y)
        # return self.crflayer(emissions, y, mask=mask)  # compare to forward, 'decode' was miss

    def forward(self, x, sent_lengths):  # 前向传播函数，模型的 input -> forward -> output
        mask = torch.ne(x, self.pad_index)
        emissions = self.lstm_forward(x, sent_lengths)
        emissions = torch.transpose(emissions, 1, 0)  # 矩阵转置为：[batch_size*sentence_length*tag_size]
        mask = torch.transpose(mask, 1, 0)
        path_score, best_paths = self.crf_layer(emissions, mask)
        tag_list = []
        for i in range(best_paths.size(0)):
            tag_list.append(best_paths[i].cpu().data.numpy()[:torch.sum(mask[i])])
        return tag_list

    def lstm_forward(self, sentence, sent_lengths):
        x = self.embedding(sentence.to(DEVICE)).to(DEVICE)  # input embedding, output x
        x = self.drop(x)
        x = pack_padded_sequence(x, sent_lengths)  # 长度对齐
        hidden = self.init_hidden(batch_size=len(sent_lengths))
        lstm_out, _ = self.lstm(x, hidden)                 # input lstm, output lstm_out. hidden 要作为一个整体输入(h0,c0)
        lstm_out, new_batch_size = pad_packed_sequence(lstm_out)

        assert torch.equal(sent_lengths, new_batch_size.to(DEVICE))
        y = self.linear1(lstm_out.to(DEVICE))  # input linear,把输出维度变为 标签 个数。
        y = self.lin_drop(y)
        y = self.linear2(y)
        return y.to(DEVICE)
