#
# 文本分类
# 思路：将英文单词转换为对应单词的整数索引和对应的标签进行训练
# https://tensorflow.google.cn/tutorials/keras/text_classification

import tensorflow as tf
import numpy as np

# 引入imdb数据集
imdb = tf.keras.datasets.imdb
# num_words=10000 保留了训练数据中最常出现的 10,000 个单词
(train_data, train_labels), (test_data,
                             test_labels) = imdb.load_data(num_words=10000)

print(train_data.shape)
print(train_labels)

# 一个映射单词到整数索引的词典
word_index = imdb.get_word_index()

# 保留第一个索引
word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key)
                           for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# 测试输出的原始文本
print(decode_review(train_data[0]))


# 对训练数据进行长度处理
# 填充数组来保证输入数据具有相同的长度，然后创建一个大小为 max_length * num_reviews 的整型张量。
# 使用能够处理此形状数据的嵌入层作为网络中的第一层
train_data = tf.keras.preprocessing.sequence.pad_sequences(
    train_data, value=word_index["<PAD>"], padding='post', maxlen=256)

test_data = tf.keras.preprocessing.sequence.pad_sequences(
    test_data, value=word_index["<PAD>"], padding='post', maxlen=256)

print(len(train_data[0]), len(train_data[1]))

# 创建模型
model = tf.keras.Sequential()
# 嵌入层
# 层采用整数编码的词汇表，并查找每个词索引的嵌入向量（embedding vector）。这些向量是通过模型训练学习到的。向量向输出数组增加了一个维度
model.add(tf.keras.layers.Embedding(10000, 16))
# 将通过对序列维度求平均值来为每个样本返回一个定长输出向量。这允许模型以尽可能最简单的方式处理变长输入。
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(16, activation='relu'))
# 使用 Sigmoid 激活函数，其函数值为介于 0 与 1 之间的浮点数，表示概率或置信度。
model.add(tf.keras.layers.Dense(2, activation='softmax'))


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])


partial_x_train = train_data[10000:]
partial_y_train = train_labels[10000:]


x_val = train_data[:10000]
y_val = train_labels[:10000]

# fit 中的 verbose
# verbose：日志显示
# verbose = 0 为不在标准输出流 输出日志信息
# verbose = 1 为输出进度条记录
# verbose = 2 为每个epoch输出一行记录
# 默认为 1

history = model.fit(partial_x_train, partial_y_train, epochs=40,
                    batch_size=512, validation_data=(x_val, y_val), verbose=1)


# evaluate 中的 verbose
# verbose：日志显示
# verbose = 0 为不在标准输出流输出日志信息
# verbose = 1 为输出进度条记录
# 注意： 只能取 0 和 1；默认为 1

results = model.evaluate(test_data,  test_labels, verbose=2)

print(results)
