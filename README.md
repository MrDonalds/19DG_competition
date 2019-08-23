# 19DG_competition
# 这是达观算法比赛的代码。要求从加密后的数据中完成命名实体识别（NER）任务。

程序采用目前普遍采用的神经网络结构：BiLstm + CRF。
在 BiLstm 前面没有加 CNN 是因为数据加密后只有数字，没有像英文单词那样的字母组合，没有前缀之类字级别的特征。

## 数据预处理
把数据预处理成下面的格式后，使用 torchtext 可以较方便地进行接下来的预处理：生成词汇、标签表，，划分 batch 等

词语——\t——类别

123      B_a
234      I_a
345      E_a
996      O
111      S_b

最后，torchtext 得到的数据集迭代器：比如 train_iter ，训练的时候直接 for item in train_iter 即可逐个 batch 训练

注意：torchtext 中对数据的格式进行处理的 Field 的编写，和根据任务不同 dataset 类型的选择。

## 模型部分
- 程序中模型：embedding -> dropout -> bilstm -> dropout -> linear -> crf

- 词向量部分是使用 主办方提供的未标注语料 训练的 300维word2vec， 比赛群里有人尝试了 bert， 效果可以提高 3-4 个百分点。
- 目前还在学习 bert，有时间的话将进行训练尝试（由于是加密语料，需要重新训练 bert，我用的机器不一定搞得定。。。）

- 论文：Lample G, Ballesteros M, et al. Neural Architectures for Named Entity Recognition. NANCL, 2016.本来是没有在 embedding层 后面加 dropout的。但是看到这篇论文里面提到了这样做效果显著，加上 dropout 试了一下，f1值确实提高2个百分点（好论文里面都是宝...）

- crf层 其实学习都就是一个转移矩阵，训练的时候因为计算f1值挺慢，思考改进办法的时候发现有人是用验证集的 loss 来选择模型的。计算 loss 会快很多。但是用loss替代f1值可行吗？看了一些 crf 里面loss的计算原理，loss = 预测路径的分数-最佳路径的分数 和 f1 有相关性但是不能代表。

## 训练部分
- 最开始实现的模型 baseline 一个学习率（lr）学到底，在 50 个epoch之后 loss 函数曲线开始很大的震荡。
解决办法就是使用 pytorch 提供的各式各样的学习率调节策略：StepLR、MultiStepLR、ReduceLROnPlateau
- 因为固定 lr 训练完150个epoch后画出的 loss 看到了有一定的梯度变化关系，就选了torch.optim.lr_scheduler.MultiStepLR
- 结果 loss 函数抖动更加平滑，f1值略有提高。

ps：在看别人代码的时候，发现有人会给模型的每一个都设置不同的学习率（做的真细...），确实可以每一层分别设置，分别用不同调节策略。。。

## 未完待续部分
模型微调，模型融合，只是上面这两项都是成形后的小工作，还是等我把 f1 提高到 0.9 再说吧

## 对于性能改进的总结
数据：使用更多的特征（BIO -> BIOES)、数据增强（剪切组合拼接）、利用好未标注语料（词向量、ELMO、BERT）
模型：结构优化（dropout、dense）、参数调节（层数少点可以防止过拟合、batch 大些训练更快、RNN不能较好实现 Batch Normalized）



