import torch
import pickle
from tqdm import tqdm
from model import BiLstmCrf, BertBiLstmCrf
from config import Config, DEVICE
from utils.log import logger
from utils.tool import data_tool
from pytorch_pretrained_bert import BertAdam

class Ner(object):

    def __init__(self, config, train_iter, dev_iter, dev_dataset):
        self.args = config
        self._model = None
        # self.best_model = None
        self.train_iter = train_iter
        self.dev_iter = dev_iter
        self.dev_dataset = dev_dataset
        self._word_vocab = config.word_vocab
        self._tag_vocab = config.tag_vocab
        if self.args.model_choose == 'bert':
            self._model = BertBiLstmCrf(args=self.args)
        else:
            self._model = BiLstmCrf(args=self.args)  # config 参数,需要传入word_vocab 以帮助词向量与现有数据的词汇表对应。

    def train(self):  # 训练模型
        # 优化器
        if self.args.model_choose == 'bert':
            optim = BertAdam(self._model.parameters(), lr=self.args.lr, warmup=self.args.warmup_proportion, t_total=65*self.args.epoch)
        else:
            optim = torch.optim.Adam(self._model.parameters(), lr=self.args.lr)  # , weight_decay=self.args.weight_decay)
        #         scheduler=torch.optim.lr_scheduler.StepLR(optim,step_size=20,gamma=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[50, 100], gamma=0.5, last_epoch=-1)
        dev_score_best = 0
        dev_score = 0
        result_dict = {'epoch': [], 'train_loss': [], 'dev_loss': [], 'dev_score': []}
        for epoch in range(self.args.epoch):
            self._model.train()  # 设置为训练模式，另外一个是 .eval() 分析模式
            # 调整学习率
            scheduler.step()
            train_loss = 0
            for item in tqdm(self.train_iter):
                self._model.zero_grad()
                item_text_sentences = item.text[0]
                item_text_lengths = item.text[1]
                loss = self._model.loss(item_text_sentences, item_text_lengths, item.tag)
                train_loss += loss.item()
                loss.backward()
                optim.step()
            # logger.info('epoch: {}, train_loss: {}'.format(epoch, train_loss))

            self._model.eval()  # 设置为分析模式
            dev_loss = 0
            for item in (self.dev_iter):
                item_text_sentences = item.text[0]
                item_text_lengths = item.text[1]
                loss = self._model.loss(item_text_sentences, item_text_lengths, item.tag)
                dev_loss += loss.item()

# logger.info('epoch: {}, train_loss:{}, dev_loss:{}'.format(epoch, train_loss, dev_loss))

            if epoch%10 == 5:  # 计算验证集的 f1 分数,  还有一种办法就是计算验证集的 loss, 选择 loss 较小的保存下来。
                dev_score = self._validata(self.dev_dataset)
            logger.info('epoch: {}, train_loss:{}, dev_loss:{}, dev_score:{}'.format(epoch, train_loss, dev_loss, dev_score))

            result_dict['epoch'].append(epoch)
            result_dict['train_loss'].append(train_loss)
            result_dict['dev_loss'].append(dev_loss)
            result_dict['dev_score'].append(dev_score)
        pickle_result = open('ner_save/res.pkl','wb')
        pickle.dump(result_dict, pickle_result)
        pickle_result.close()

# # save checkpoints, 将分数最高的模型的参数保存下来。
# if dev_score > dev_score_best:
#     dev_score_best = dev_score
#     checkpoint = {
#         'state_dict': self._model.state_dict(),
#         'config': self.args
#     }
#     torch.save(checkpoint, '{}checkpoint_{}'.format(epoch, dev_score_best))
#     logger.info('Best tmp model f1: {}'.format(dev_score_best))
# else:
#     # 要加载checkpoint的话：self._model.load_state_dict(torch.load('checkpoint')['state_dict'])
#     pass  # 可以选择在此处调整学习率

    def predict(self, text):  # 得到预测结果
        self._model.eval()
        vec_text = torch.tensor([self._word_vocab.stoi[x] for x in text])
        len_text = torch.tensor([len(vec_text)]).to(DEVICE)
        vec_predict = self._model(vec_text.view(-1, 1).to(DEVICE), len_text)[0]
        tag_predict = [self._tag_vocab.itos[i] for i in vec_predict]
        res = self.convert(text, tag_predict)
        return res

    def save(self):
        self.args.save()
        self._model.save()

    def load(self, save_path='ner_save'):  # 加载配置参数和模型
        config = Config.load(save_path)
        self._model = BiLstmCrf(config)
        self._model.load()
        self._word_vocab = config.word_vocab
        self._tag_vocab = config.tag_vocab
        pass

    def test(self):
        pass

    def _validata(self, dev_dataset):  # 验证模型，返回 f1 分数
        self._model.eval()
        dev_correct_list = []
        dev_total_p_list = []
        dev_total_t_list = []
        for dev_item in dev_dataset:
            correct, total_p, total_t = data_tool.get_score(self._model, dev_item.text, dev_item.tag, self._word_vocab,
                                                            self._tag_vocab)
            dev_correct_list.append(correct)
            dev_total_p_list.append(total_p)
            dev_total_t_list.append(total_t)
        p = sum(dev_correct_list) / sum(dev_total_p_list) if sum(dev_total_p_list) != 0 else 0
        r = sum(dev_correct_list) / sum(dev_total_t_list) if sum(dev_total_t_list) != 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) != 0 else 0  # 拿整个数据集的 p 和 r 计算 f1
        return f1

    @staticmethod
    def convert(text, tags):  # 把单个的预测组合起来，转换成对词组的预测，系统的评价也是对词组是否正确进行预测的。
        ranges = []
        begin = 0
        for i, tag in enumerate(tags):
            if tag.split('_')[0] == 'O':
                if i == len(tags) - 1 or tags[i + 1].split('_')[0] != 'O':
                    ranges.append('_'.join(text[begin: i + 1]) + '/o')
                    begin = i + 1
            elif tag.split('_')[0] == 'B':
                # begin = i
                temp_type = tag.split('_')[1]
                if i == len(tags) - 1 or (tags[i + 1].split('_')[0] != 'I' and tags[i + 1].split('_')[0] != 'E'):
                    ranges.append('_'.join(text[begin: i + 1]) + '/' + temp_type)
                    begin = i + 1
            elif tag.split('_')[0] == 'I':
                pass
                # temp_type = tag.split('_')[1]
                # if i == len(tags) - 1 or tags[i + 1].split('_')[0] != 'I':
                #     ranges.append('_'.join(tags[begin: i + 1]) + '/' + temp_type)
                #     begin = i + 1
            elif tag.split('_')[0] == 'E':
                temp_type = tag.split('_')[1]
                ranges.append('_'.join(text[begin: i + 1]) + '/' + temp_type)
                begin = i + 1
            elif tag.split('_')[0] == 'S':
                temp_type = tag.split('_')[1]
                ranges.append('_'.join(text[begin: i + 1]) + '/' + temp_type)
                begin = i + 1
            else:
                print('error: **********************************')
        return '  '.join(ranges)


class Ner_bert(object):
    def __init__(self, config, train_iter, dev_iter, dev_dataset):
        self.args = config
        self._model = None
        # self.best_model = None
        self.train_iter = train_iter
        self.dev_iter = dev_iter
        self.dev_dataset = dev_dataset
        self._word_vocab = config.word_vocab
        self._tag_vocab = config.tag_vocab

        self._model = BiLstmCrf(args=self.args)  # config 参数,需要传入word_vocab 以帮助词向量与现有数据的词汇表对应。

    def train(self):
        optimizer = BertAdam(self._model.parameters, lr=self.args.lr)


