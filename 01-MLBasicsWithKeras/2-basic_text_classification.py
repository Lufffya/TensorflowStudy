#! -*- coding:utf-8 -*-
# 电影评论文本分类
# 使用评论文本将影评分为积极（positive）或消极（nagetive）两类
# 这是一个二元（binary）或者二分类问题, 一种重要且应用广泛的机器学习问题
# https://tensorflow.google.cn/tutorials/keras/text_classification

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# IMDB 数据集已经打包在 Tensorflow 中. 该数据集已经经过预处理, 评论（单词序列）已经被转换为整数序列, 其中每个整数表示字典中的特定单词
imdb = tf.keras.datasets.imdb

# 参数 num_words=10000 保留了训练数据中最常出现的 10,000 个单词. 为了保持数据规模的可管理性, 低频词将被丢弃
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# 每个样本都是一个表示影评中词汇的整数数组. 每个标签都是一个值为 0 或 1 的整数值, 其中 0 代表消极评论, 1 代表积极评论
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# 评论文本被转换为整数值, 其中每个整数代表词典中的一个单词. 首条评论是这样的
print(train_data[0])

# 电影评论可能具有不同的长度. 以下代码显示了第一条和第二条评论的中单词数量. 由于神经网络的输入必须是统一的长度, 我们稍后需要解决这个问题
print(len(train_data[0]), len(train_data[1]))

# 一个映射单词到整数索引的词典
word_index = imdb.get_word_index()

# 保留第一个索引
word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# 现在我们可以使用 decode_review 函数来显示首条评论的文本
print(decode_review(train_data[0]))

# 影评——即整数数组必须在输入神经网络之前转换为张量. 这种转换可以通过以下两种方式来完成
# 将数组转换为表示单词出现与否的由 0 和 1 组成的向量, 类似于 one-hot 编码. 
# 例如: 序列[3, 5]将转换为一个 10,000 维的向量, 该向量除了索引为 3 和 5 的位置是 1 以外, 其他都为 0 . 然后, 将其作为网络的首层——一个可以处理浮点型向量数据的稠密层. 不过, 这种方法需要大量的内存, 需要一个大小为 num_words * num_reviews 的矩阵
# 或者, 我们可以填充数组来保证输入数据具有相同的长度, 然后创建一个大小为 max_length * num_reviews 的整型张量. 我们可以使用能够处理此形状数据的嵌入层作为网络中的第一层
# 在本教程中, 我们将使用第二种方法
# 由于电影评论长度必须相同, 我们将使用 pad_sequences 函数来使长度标准化
train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding='post', maxlen=256)

test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=256)

# 现在让我们看下样本的长度
print(len(train_data[0]), len(train_data[1]))

# 构建模型
# 神经网络由堆叠的层来构建, 这需要从两个主要方面来进行体系结构决策
# 模型里有多少层?
# 每个层里有多少隐层单元（hidden units）?
# 在此样本中, 输入数据包含一个单词索引的数组. 要预测的标签为 0 或 1. 让我们来为该问题构建一个模型
# 输入形状是用于电影评论的词汇数目（10,000 词）
vocab_size = 10000

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 16))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary()

# 层按顺序堆叠以构建分类器：
# 1.第一层是嵌入（Embedding）层. 该层采用整数编码的词汇表, 并查找每个词索引的嵌入向量（embedding vector）
# 这些向量是通过模型训练学习到. 向量向输出数组增加了一个维度. 得到的维度为：(batch, sequence, embedding)
# 2.接下来, GlobalAveragePooling1D 将通过对序列维度求平均值来为每个样本返回一个定长输出向量. 这允许模型以尽可能最简单的方式处理变长输入
# 3.该定长输出向量通过一个有 16 个隐层单元的全连接（Dense）层传输
# 4.最后一层与单个输出结点密集连接. 使用 Sigmoid 激活函数, 其函数值为介于 0 与 1 之间的浮点数, 表示概率或置信度

# 损失函数与优化器
# 一个模型需要损失函数和优化器来进行训练. 由于这是一个二分类问题且模型输出概率值（一个使用 sigmoid 激活函数的单一单元层）, 我们将使用 binary_crossentropy 损失函数
# 这不是损失函数的唯一选择, 例如, 您可以选择 mean_squared_error 
# 但是, 一般来说 binary_crossentropy 更适合处理概率——它能够度量概率分布之间的“距离”, 或者在我们的示例中, 指的是度量 ground-truth 分布与预测值之间的“距离”
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# 创建一个验证集
# 在训练时, 我们想要检查模型在未见过的数据上的准确率（accuracy）
# 通过从原始训练数据中分离 10,000 个样本来创建一个验证集.（为什么现在不使用测试集？我们的目标是只使用训练数据来开发和调整模型, 然后只使用一次测试数据来评估准确率（accuracy））

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]


# 训练模型
# 以 512 个样本的 mini-batch 大小迭代 40 个 epoch 来训练模型。这是指对 x_train 和 y_train 张量中所有样本的的 40 次迭代
# 在训练过程中, 监测来自验证集的 10,000 个样本上的损失值（loss）和准确率（accuracy）

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val), verbose=1)
# fit 中的 verbose
# verbose：日志显示
# verbose = 0 为不在标准输出流 输出日志信息
# verbose = 1 为输出进度条记录
# verbose = 2 为每个epoch输出一行记录
# 默认为 1


# 评估模型
# 我们来看一下模型的性能如何. 将返回两个值. 损失值（loss）（一个表示误差的数字, 值越低越好）与准确率（accuracy）
results = model.evaluate(test_data,  test_labels, verbose=2)

print(results)

# evaluate 中的 verbose
# verbose：日志显示
# verbose = 0 为不在标准输出流输出日志信息
# verbose = 1 为输出进度条记录
# 注意： 只能取 0 和 1;默认为 1


# 创建一个准确率（accuracy）和损失值（loss）随时间变化的图表
# model.fit() 返回一个 History 对象, 该对象包含一个字典, 其中包含训练阶段所发生的一切事件：
history_dict = history.history
history_dict.keys()


acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# “bo”代表 "蓝点"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b代表“蓝色实线”
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


plt.clf()   # 清除数字

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# 在该图中, 点代表训练损失值（loss）与准确率（accuracy）, 实线代表验证损失值（loss）与准确率（accuracy）

# 注意训练损失值随每一个 epoch 下降而训练准确率（accuracy）随每一个 epoch 上升。这在使用梯度下降优化时是可预期的——理应在每次迭代中最小化期望值

# 验证过程的损失值（loss）与准确率（accuracy）的情况却并非如此——它们似乎在 20 个 epoch 后达到峰值。这是过拟合的一个实例：模型在训练数据上的表现比在以前从未见过的数据上的表现要更好. 在此之后, 模型过度优化并学习特定于训练数据的表示, 而不能够泛化到测试数据

# 对于这种特殊情况, 我们可以通过在 20 个左右的 epoch 后停止训练来避免过拟合. 稍后, 您将看到如何通过回调自动执行此操作
