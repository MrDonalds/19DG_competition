import time
import torch
from utils.tool import data_tool
from data import data_2iob
from config import DEFAULT_CONFIG, Config
from module import Ner

# dataset_path = 'train.txt'
# testset_path = 'test.txt'
# train_path = 'data/train_dev_ner_iobes.txt'
# dev_path = 'data/train_dev_ner_iobes.txt'
save_path = './ner_save'
vectors_path = 'data/corpus_300'


def main():
    start = time.time()
    print("读取训练数据...")
    # train_path = data_2iob.format_data(dataset_path)  # 由datagrand中的两个函数生成规则的训练数据集
    for i in range(4):
        train_path = 'data/dev_data/train_k{}_ner_iobes.txt'.format(i)
        dev_path = 'data/dev_data/validate_k{}_ner_iobes.txt'.format(i)

        train_dataset = data_tool.get_dataset(train_path)
        dev_dataset = data_tool.get_dataset(dev_path)
        word_vocab, tag_vocab = data_tool.get_vocab(train_dataset, dev_dataset)

        train_iter = data_tool.get_iterator(train_dataset, batch_size=DEFAULT_CONFIG['batch_size'])
        config = Config(word_vocab, tag_vocab, vector_path=vectors_path, save_path=save_path)

        model = Ner(config, train_iter, dev_dataset)

        model.train()

        # model.save()

        model.predict()  # 得到测试集的预测结果，生成可提交的文件

        end = time.time()
        print('{} total run time: '.format(i), int(end-start))


if __name__ == '__main__':
    main()
