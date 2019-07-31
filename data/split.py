import numpy as np
from sklearn.model_selection import KFold
from data import data_2iobes

train_numpy = []
with open('./datagrand/train.txt') as files:
    for file in files:
        train_numpy.append(file)
train_numpy = np.array(train_numpy)


kf = KFold(n_splits=5, shuffle=False, random_state=42)  # K折交叉检验，5等分，划分时不进行洗牌(每次划分结果相同)
for i, (train_idx, test_idx) in enumerate(kf.split(train_numpy)):  # Split 划分数据集(训练集、测试集) i=0,1,2,3,4
    X_train = train_numpy[train_idx]
    X_test = train_numpy[test_idx]
    with open('./dev_data/train_k{}.txt'.format(i), 'w') as f:  # 分别保存划分好的数据集， 放在 original data
        for t in X_train:
            f.write(t)
    with open('./dev_data/validate_k{}.txt'.format(i), 'w') as f:
        for t in X_test:
            f.write(t)

for i in range(5):
    path = './dev_data/train_k{}.txt'.format(i)
    data_2iobes.raw_to_iobes(path)
    path = './dev_data/validate_k{}.txt'.format(i)
    data_2iobes.raw_to_iobes(path)
