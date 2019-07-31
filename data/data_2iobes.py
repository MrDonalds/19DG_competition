import math
import random
from data import data_2iob

def data_format_iobes(data_path):
    print('format file into iobes: ', data_path)
    with open(data_path, 'r') as f:
        data_iobes_path = data_path.split('.')[0]+'_iobes'+'.txt'
        with open(data_iobes_path, 'w') as fw:
            stences = []
            for line in f:
                if line.strip():
                    stences.append(line.strip())
                else:
                    stences = iob_iobes(stences)
                    fw.write('\n'.join(stences) + '\n')
                    fw.write('\n')
                    stences = []
    print('format iobes into: ', data_iobes_path)
    return data_iobes_path


def iob_iobes(stences):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, line in enumerate(stences):
        tag = line.split('\t')[1]
        if tag == 'O':
            new_tags.append(line.split('\t')[0]+'\t'+tag)
        elif tag.split('_')[0] == 'B':
            if i + 1 != len(stences) and stences[i + 1].split('\t')[1].split('_')[0] == 'I':
                new_tags.append(line.split('\t')[0]+'\t'+tag)
            else:
                new_tags.append(line.split('\t')[0]+'\t'+tag.replace('B_', 'S_'))
        elif tag.split('_')[0] == 'I':
            if i + 1 < len(stences) and stences[i + 1].split('\t')[1].split('_')[0] == 'I':
                new_tags.append(line.split('\t')[0]+'\t'+tag)
            else:
                new_tags.append(line.split('\t')[0]+'\t'+tag.replace('I_', 'E_'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def raw_to_iobes(path='./datagrand/train.txt'):
    data_path = data_2iob.format_data(path)
    data_format_iobes(data_path)


# raw_to_iobes()











class BatchManager(object):
    def __init__(self, data,  batch_size):
        # 排序并填充，使单个批次的每个样本保持长度一致，不同批次的长度不一定相同
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) /batch_size))
        # 按句子长度进行排序
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i*int(batch_size) : (i+1)*int(batch_size)]))
        return batch_data

    @staticmethod
    def pad_data(data):
        strings = []
        chars = []
        segs = []
        targets = []
        intents = []
        max_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            string, char, seg, target, intent = line
            padding = [0] * (max_length - len(string))
            strings.append(string + padding)
            chars.append(char + padding)
            segs.append(seg + padding)
            targets.append(target + padding)
            intents.append(intent + padding)

        return [strings, chars, segs, targets, intents]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]

