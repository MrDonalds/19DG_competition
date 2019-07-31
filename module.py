import torch
from tqdm import tqdm
from model import BiLstmCrf
from config import Config
from utils.log import logger
from utils.tool import data_tool


class Ner(object):

    def __init__(self, config, train_iter, dev_dataset):
        self.args = config
        self._model = None
        # self.best_model = None
        self.train_iter = train_iter
        self.dev_dataset = dev_dataset
        self._word_vocab = config.word_vocab
        self._tag_vocab = config.tag_vocab

        self._model = BiLstmCrf(args=self.args)  # config 参数,需要传入word_vocab 以帮助词向量与现有数据的词汇表对应。

    def train(self):  # 训练模型
        # 优化器
        optim = torch.optim.Adam(self._model.parameters(), lr=self.args.lr)  #, weight_decay=self.args.weight_decay)

        dev_score_best = 0
        for epoch in range(self.args.epoch):
            self._model.train()  # 设置为训练模式，另外一个是 .eval() 分析模式
            acc_loss = 0
            for item in tqdm(self.train_iter):
                self._model.zero_grad()
                item_text_sentences = item.text[0]
                item_text_lengths = item.text[1]
                item_loss = (-self._model.loss(item_text_sentences, item_text_lengths, item.tag)) / item.tag.size(1)
                acc_loss += item_loss.view(-1).cpu().data.tolist()[0]
                item_loss.backward()
                optim.step()
            logger.info('epoch: {}, acc_loss: {}'.format(epoch, acc_loss))

            # 计算验证集的 f1 分数,  还有一种办法就是计算验证集的 loss, 选择 loss 较小的保存下来。
            dev_score = self._validata(self.dev_dataset)
            logger.info('dev score:{}'.format(dev_score))

            # save checkpoints, 将分数最高的模型的参数保存下来。
            if dev_score > dev_score_best:
                dev_score_best = dev_score
                checkpoint = {
                    'state_dict': self._model.state_dict(),
                    'config': self.args
                }
                torch.save(checkpoint, '{}checkpoint_{}'.format(epoch, dev_score_best))
                logger.info('Best tmp model f1: {}'.format(dev_score_best))
            else:
                # 要加载checkpoint的话：self._model.load_state_dict(torch.load('checkpoint')['state_dict'])
                pass  # 可以选择在此处调整学习率

    def predict(self):  # 得到预测结果
        pass

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
        for dev_item in tqdm(dev_dataset):
            correct, total_p, total_t = data_tool.get_score(self._model, dev_item.text, dev_item.tag, self._word_vocab,
                                                           self._tag_vocab)
            dev_correct_list.append(correct)
            dev_total_p_list.append(total_p)
            dev_total_t_list.append(total_t)
        p = sum(dev_correct_list) / sum(dev_total_p_list) if sum(dev_total_p_list) != 0 else 0
        r = sum(dev_correct_list) / sum(dev_total_t_list) if sum(dev_total_t_list) != 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) != 0 else 0  # 拿整个数据集的 p 和 r 计算 f1
        return f1
