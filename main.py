import time
import torch
from utils.tool import data_tool
from data import data_2iob
from config import DEFAULT_CONFIG, Config
from module import Ner
import numpy as np

# dataset_path = 'train.txt'
# testset_path = 'test.txt'
# train_path = 'data/train_dev_ner_iobes.txt'
# dev_path = 'data/train_dev_ner_iobes.txt'
save_path = './ner_save'
vectors_path = 'data/corpus_300'


def main():
    start = time.time()
    print("读取训练数据...")
    #     train_path = data_2iob.format_data(dataset_path)  # 由datagrand中的两个函数生成规则的训练数据集

    # for i in range(5):  # 5折交叉验证
    i = 0
    train_path = 'data/dev_data/train_k{}_ner_iobes.txt'.format(i)
    dev_path = 'data/dev_data/validate_k{}_ner_iobes.txt'.format(i)

    train_dataset = data_tool.get_dataset(train_path)
    dev_dataset = data_tool.get_dataset(dev_path)
    word_vocab, tag_vocab = data_tool.get_vocab(train_dataset, dev_dataset)

    train_iter = data_tool.get_iterator(train_dataset, batch_size=DEFAULT_CONFIG['batch_size'])
    dev_iter = data_tool.get_iterator(dev_dataset, batch_size=DEFAULT_CONFIG['batch_size'])
    config = Config(word_vocab, tag_vocab, vector_path=vectors_path, save_path=save_path)

    model = Ner(config, train_iter, dev_iter, dev_dataset)

    model.train()

    # ####################### 全部数据集一起训练得到模型
    # train_path = 'train_ner_iobes.txt'
    # train_dataset = data_tool.get_dataset(train_path)
    # word_vocab, tag_vocab = data_tool.get_vocab(train_dataset)
    # train_iter = data_tool.get_iterator(train_dataset, batch_size=DEFAULT_CONFIG['batch_size'])
    #
    # config = Config(word_vocab, tag_vocab, vector_path=vectors_path, save_path=save_path)
    #
    # model = Ner(config, train_iter, None, None)

    # model.train()

    model.save()

    # ####################### 加载模型进行预测，生成提交文件
    # config = Config(None, None, vector_path=vectors_path, save_path=save_path)
    # model = Ner(config, None, None, None)
    model.load()
    with open('./data/datagrand/test.txt', 'r') as f:
        with open('test_sumbit.txt', 'w') as sub:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip().split('_')
                res = model.predict(line)
                if i % 1000 == 0:
                    print(i)
                sub.write(res + '\n')

    end = time.time()
    print('total run time: ', int(end - start))


if __name__ == '__main__':
    seed = 2019
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    main()
