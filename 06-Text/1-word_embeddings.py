#! -*- coding:utf-8 -*-

import io
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization


# 下载 IMDb 数据集
# 您将在本教程中使用大型电影评论数据集.
# 您将在此数据集上训练情感分类器模型, 并在此过程中从头开始学习嵌入. 要阅读有关从头开始加载数据集的更多信息, 请参阅加载文本教程.

# 使用 Keras 文件实用程序下载数据集并查看目录
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url, untar=True, cache_dir='.', cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
print(os.listdir(dataset_dir))

# 看一下train/目录.
# 它具有分别标记为正面pos和neg负面的电影评论的文件夹. 您将使用来自pos和neg文件夹的评论来训练二元分类模型.
train_dir = os.path.join(dataset_dir, 'train')
print(os.listdir(train_dir))

# 该train目录还有其他文件夹, 应在创建训练数据集之前将其删除.
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

# 接下来, 创建一个tf.data.Dataset使用tf.keras.utils.text_dataset_from_directory. 您可以在此文本分类教程中阅读有关使用此实用程序的更多信息.

# 使用该train目录创建训练数据集和验证数据集, 拆分为 20% 以进行验证.
batch_size = 1024
seed = 123
train_ds = tf.keras.utils.text_dataset_from_directory('aclImdb/train', batch_size=batch_size, validation_split=0.2, subset='training', seed=seed)
val_ds = tf.keras.utils.text_dataset_from_directory('aclImdb/train', batch_size=batch_size, validation_split=0.2, subset='validation', seed=seed)

# 看看(1: positive, 0: negative)火车数据集中的一些电影评论及其标签.
for text_batch, label_batch in train_ds.take(1):
  for i in range(5):
    print(label_batch[i].numpy(), text_batch.numpy()[i])

# 配置数据集以提高性能
# 这是加载数据时应使用的两种重要方法, 以确保 I/O 不会阻塞.

# .cache()将数据从磁盘加载后保留在内存中. 这将确保数据集在训练模型时不会成为瓶颈. 
# 如果您的数据集太大而无法放入内存, 您还可以使用此方法创建一个高性能的磁盘缓存, 这比许多小文件的读取效率更高.
# .prefetch()在训练时重叠数据预处理和模型执行.

# 您可以在数据性能指南中了解有关这两种方法以及如何将数据缓存到磁盘的更多信息.
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 使用嵌入层
# Keras 使词嵌入的使用变得容易. 看一下嵌入层.

# Embedding 层可以理解为从整数索引（代表特定单词）映射到密集向量（它们的嵌入）的查找表.
# 嵌入的维数（或宽度）是一个参数, 您可以试验以查看哪种方法对您的问题有效, 这与试验密集层中神经元数量的方式非常相似.

# Embed a 1,000 word vocabulary into 5 dimensions.
embedding_layer = tf.keras.layers.Embedding(1000, 5)

# 当您创建一个嵌入层时, 嵌入的权重是随机初始化的（就像任何其他层一样）.
# 在训练期间, 它们通过反向传播逐渐调整. 一旦训练, 学习的词嵌入将粗略地编码词之间的相似性（因为它们是针对您的模型训练的特定问题而学习的）.

# 如果将整数传递给嵌入层, 则结果会将每个整数替换为嵌入表中的向量:
result = embedding_layer(tf.constant([1, 2, 3]))
print(result.numpy())

# 对于文本或序列问题, 嵌入层采用整数的二维张量, 形状为(samples, sequence_length), 其中每个条目都是整数序列.
# 它可以嵌入可变长度的序列. (32, 10)您可以将形状（长度为 10 的 32 个序列的批次）或(64, 15)（长度为 15 的 64 个序列的批次）输入到嵌入层之上.

# 返回的张量比输入多一个轴, 嵌入向量沿新的最后一个轴对齐. 将输入批次传递给它(2, 3), 输出为(2, 3, N)
result = embedding_layer(tf.constant([[0, 1, 2], [3, 4, 5]]))
print(result.shape)

# 当给定一批序列作为输入时, 嵌入层返回一个形状为 的 3D 浮点张量(samples, sequence_length, embedding_dimensionality). 
# 要将这个可变长度序列转换为固定表示, 有多种标准方法. 
# 在将其传递给密集层之前, 您可以使用 RNN、注意力或池化层. 
# 本教程使用池化, 因为它是最简单的. 带有 RNN的文本分类教程是一个很好的下一步.

# 文本预处理
# 接下来, 定义情绪分类模型所需的数据集预处理步骤. 
# 使用所需参数初始化 TextVectorization 层以矢量化电影评论. 您可以在文本分类教程中了解有关使用此图层的更多信息.
# Create a custom standardization function to strip HTML break tags '<br />'.
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')


# Vocabulary size and number of words in a sequence.
vocab_size = 10000
sequence_length = 100

# Use the text vectorization layer to normalize, split, and map strings to
# integers. Note that the layer uses the custom standardization defined above.
# Set maximum_sequence length as all samples are not of the same length.
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (no labels) and call adapt to build the vocabulary.
text_ds = train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)

# 创建分类模型
# 使用Keras Sequential API定义情感分类模型. 在这种情况下, 它是“连续词袋”样式模型.
# 1, 该TextVectorization层将字符串转换为词汇索引.
#   您已经初始化vectorize_layer为 TextVectorization 层并通过调用构建其词汇表adapt.
#   text_ds现在 vectorize_layer 可以用作端到端分类模型的第一层, 将转换后的字符串输入嵌入层.
# 2, 该Embedding层采用整数编码的词汇表并查找每个单词索引的嵌入向量.
#   这些向量是作为模型训练来学习的. 向量向输出数组添加一个维度. 结果尺寸为: (batch, sequence, embedding).
# 3, 该GlobalAveragePooling1D层通过对序列维度进行平均来为每个示例返回一个固定长度的输出向量. 这允许模型以最简单的方式处理可变长度的输入.
# 4, Dense固定长度的输出向量通过具有 16 个隐藏单元的全连接 ( ) 层进行管道传输.
# 5, 最后一层与单个输出节点紧密连接.
embedding_dim=16

model = Sequential([
    vectorize_layer,
    Embedding(vocab_size, embedding_dim, name="embedding"),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1)
])

# 编译和训练模型
# 您将使用TensorBoard可视化指标, 包括损失和准确性. 创建一个tf.keras.callbacks.TensorBoard.
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

# Adam使用优化器和BinaryCrossentropy损失编译和训练模型.
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[])

print(model.summary())
