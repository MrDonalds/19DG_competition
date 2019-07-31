import torch
import numpy as np
from torchtext.data import Dataset, Field, BucketIterator, ReversibleField
from torchtext.vocab import Vectors
from torchtext.datasets import SequenceTaggingDataset
from utils.log import logger
from config import DEVICE, DEFAULT_CONFIG


def light_tokenize(sequence: str):
    return [sequence]


# 普通字段
TEXT = Field(sequential=True, tokenize=light_tokenize, include_lengths=True)
# 可逆字段：可以从 wordid 映射回原来的 word
TAG = ReversibleField(sequential=True, tokenize=light_tokenize, is_target=True, unk_token=None)
Fields = [('text', TEXT), ('tag', TAG)]


class TOOL(object):

    # @staticmethod
    def get_dataset(self, path: str, fields=Fields, separator='\t'):
        logger.info('loading dataset from {}'.format(path))
        st_dataset = SequenceTaggingDataset(path, fields=fields, separator=separator)
        logger.info('successed loading dataset')
        return st_dataset

    def get_vocab(self, *dataset):
        logger.info('building word vocab...')
        TEXT.build_vocab(*dataset)
        logger.info('successed building word vocab')
        logger.info('building tag vocab...')
        TAG.build_vocab(*dataset)
        logger.info('successed building tag vocab')
        return TEXT.vocab, TAG.vocab

    def get_vectors(self, path: str):
        logger.info('loading vectors from {}'.format(path))
        vectors = Vectors(path)
        logger.info('successed loading vectors')
        return vectors

    def get_iterator(self, dataset: Dataset, batch_size=DEFAULT_CONFIG['batch_size'], device=DEVICE,
                     sort_key=lambda x: len(x.text), sort_within_batch=True):
        return BucketIterator(dataset, batch_size=batch_size, device=device, sort_key=sort_key,
                              sort_within_batch=sort_within_batch)  # 会打乱顺序，把长度相似的排在一起

    def get_score(self, model, text, tag, word_vocab, tag_vocab, score_type='f1'):

        vec_x = torch.tensor([word_vocab.stoi[i] for i in text])  # 将文本转换为整数
        len_vec_x = torch.tensor([len(vec_x)]).to(DEVICE)   # 文本长度

        predict_y = model(vec_x.view(-1, 1).to(DEVICE), len_vec_x)[0]  # 得到模型的预测值
        true_y = [tag_vocab.stoi[i] for i in tag]  # 真实标签
        assert len(true_y) == len(predict_y)

        return self.trasform_f1(predict_y, true_y, tag_vocab)

    def trasform_f1(self, pre, tru, tag_vac):
        dit = {}
        total_p = 0
        total_t = 0
        for key in tag_vac.stoi:
            dit[tag_vac.stoi[key]] = key
        true_tag = [dit[i] for i in tru]
        pred_tag = [dit[i] for i in pre]
        true_tag = self.convert(true_tag)
        pred_tag = self.convert(pred_tag)
        # for t, p in zip(true_tag, pred_tag):
        correct = sum(1 for i in pred_tag if i in true_tag) if pred_tag != {''} else 0
        total_p += len(pred_tag) if pred_tag != {''} else 0
        total_t += len(true_tag) if true_tag != {''} else 0
        # correct.append(c)
        # p = correct / total_p if total_p != 0 else 0  # 不计算单个句子的 f1
        # r = correct / total_t if total_t != 0 else 0
        # f1 = 2 * p * r / (p + r) if (p + r) != 0 else 0

        return correct, total_p, total_t

    def convert(self, tags):  # 把单个的预测组合起来，转换成对词组的预测，系统的评价也是对词组是否正确进行预测的。
        ranges = []
        begin = 0
        for i, tag in enumerate(tags):
            if tag.split('_')[0] == 'O':
                if i == len(tags) - 1 or tags[i + 1].split('_')[0] != 'O':
                    ranges.append('_'.join(tags[begin: i + 1]) + '/o')
                    begin = i + 1
            elif tag.split('_')[0] == 'B':
                # begin = i
                temp_type = tag.split('_')[1]
                if i == len(tags) - 1 or (tags[i + 1].split('_')[0] != 'I' and tags[i + 1].split('_')[0] != 'E'):
                    ranges.append('_'.join(tags[begin: i + 1]) + '/' + temp_type)
                    begin = i + 1
            elif tag.split('_')[0] == 'I':
                pass
                # temp_type = tag.split('_')[1]
                # if i == len(tags) - 1 or tags[i + 1].split('_')[0] != 'I':
                #     ranges.append('_'.join(tags[begin: i + 1]) + '/' + temp_type)
                #     begin = i + 1
            elif tag.split('_')[0] == 'E':
                temp_type = tag.split('_')[1]
                ranges.append('_'.join(tags[begin: i + 1]) + '/' + temp_type)
                begin = i + 1
            elif tag.split('_')[0] == 'S':
                temp_type = tag.split('_')[1]
                ranges.append('_'.join(tags[begin: i + 1]) + '/' + temp_type)
                begin = i + 1
            else:
                print('error: **********************************')
        return ranges

data_tool = TOOL()

