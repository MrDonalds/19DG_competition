from torchtext.vocab import Vectors
import pickle

def format_data(dataset_path):
    print('begin {}'.format(dataset_path))
    with open(dataset_path) as f:
        format_path = dataset_path.split('.')[1].split('/')[-1]+'_ner'+'.txt'
        with open(format_path, 'w') as fw:
            for i, sentence in enumerate(f):
                delimiter = '\t'
                words = sentence.replace('\n', '').split('  ')  # 将一段文本末尾的换行符去掉，然后按两个空格切分(对于训练集的切分)。
                for j, word in enumerate(words):
                    split_word = word.split('/')  # 训练集里面才有‘/’,前面是文本后面是标签abc。对于测试集split_word不变
                    if 'train' == 'train':
                        tag = split_word[1]
                    else:
                        tag = 'O'
                    word_meta = split_word[0]
                    word_meta_split = word_meta.split('_')  # 划分为单词。
                    meta_len = len(word_meta_split)
                    if tag == 'a':
                        if meta_len == 1:
                            fw.write(word_meta_split[0] + delimiter + 'B_a' + '\n')
                            # fw.write(word_meta_split[0] + delimiter + 'W_a' + '\n')
                            # fw.write(word_meta_split[0] + delimiter + 'O' + '\n')
                        else:
                            for k, char in enumerate(word_meta_split):
                                if k == 0:
                                    fw.write(char + delimiter + 'B_a' + '\n')
                                elif k == meta_len - 1:
                                    # fw.write(char + delimiter + 'E_a' + '\n')
                                    fw.write(char + delimiter + 'I_a' + '\n')
                                else:
                                    # fw.write(char + delimiter + 'M_a' + '\n')
                                    fw.write(char + delimiter + 'I_a' + '\n')
                    elif tag == 'b':
                        if meta_len == 1:
                            fw.write(word_meta_split[0] + delimiter + 'B_b' + '\n')
                            # fw.write(word_meta_split[0] + delimiter + 'W_b' + '\n')
                            # fw.write(word_meta_split[0] + delimiter + 'O' + '\n')
                        else:
                            for k, char in enumerate(word_meta_split):
                                if k == 0:
                                    fw.write(char + delimiter + 'B_b' + '\n')
                                elif k == meta_len - 1:
                                    fw.write(char + delimiter + 'I_b' + '\n')
                                else:
                                    # fw.write(char + delimiter + 'M_b' + '\n')
                                    fw.write(char + delimiter + 'I_b' + '\n')
                    elif tag == 'c':
                        if meta_len == 1:
                            fw.write(word_meta_split[0] + delimiter + 'B_c' + '\n')
                            # fw.write(word_meta_split[0] + delimiter + 'W_c' + '\n')
                            # fw.write(word_meta_split[0] + delimiter + 'O' + '\n')
                        else:
                            for k, char in enumerate(word_meta_split):
                                if k == 0:
                                    fw.write(char + delimiter + 'B_c' + '\n')
                                elif k == meta_len - 1:
                                    fw.write(char + delimiter + 'I_c' + '\n')
                                else:
                                    # fw.write(char + delimiter + 'M_c' + '\n')
                                    fw.write(char + delimiter + 'I_c' + '\n')
                    else:
                        if meta_len == 1:
                            fw.write(word_meta_split[0] + delimiter + 'O' + '\n')
                        else:
                            for k, char in enumerate(word_meta_split):
                                fw.write(char + delimiter + 'O' + '\n')
                fw.write('\n')  # 每一段话隔一个空行，crf 的要求。
    print('finish {}'.format(format_path))
    return format_path


# vectors_path = 'corpus_300.txt'
#
# V = Vectors(vectors_path, cache='data')
#
# with open('corpus_300', 'wb') as f:
#     pickle.dump(V, f)
