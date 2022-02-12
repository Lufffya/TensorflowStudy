#! -*- coding:utf-8 -*-
# 使用 Keras 和 Tensorflow Hub 对电影评论进行文本分类
# 本教程演示了使用 Tensorflow Hub 和 Keras 进行迁移学习的基本应用。
# TensorFlow Hub, 一个用于迁移学习的库和平台
# https://tensorflow.google.cn/tutorials/keras/text_classification_with_hub


import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")


# 将训练集分割成 60% 和 40%, 从而最终我们将得到 15,000 个训练样本
# 10,000 个验证样本以及 25,000 个测试样本。
train_data, validation_data, test_data = tfds.load(name="imdb_reviews", split=('train[:60%]', 'train[60%:]', 'test'), as_supervised=True)

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
print(train_examples_batch)
print(train_labels_batch)

# 表示文本的一种方式是将句子转换为嵌入向量（embeddings vectors）. 我们可以使用一个预先训练好的文本嵌入（text embedding）作为首层, 这将具有三个优点：
# 我们不必担心文本预处理
# 我们可以从迁移学习中受益
# 嵌入具有固定长度, 更易于处理
# 针对此示例我们将使用 TensorFlow Hub 中名为 google/tf2-preview/gnews-swivel-20dim/1 的一种预训练文本嵌入（text embedding）模型 

# 让我们首先创建一个使用 Tensorflow Hub 模型嵌入（embed）语句的Keras层, 并在几个输入样本中进行尝试
# 请注意无论输入文本的长度如何, 嵌入（embeddings）输出的形状都是：(num_examples, embedding_dimension)

embedding = "https://hub.tensorflow.google.cn/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)

hub_layer(train_examples_batch[:3])

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.summary()

# 层按顺序堆叠以构建分类器：
# 第一层是 Tensorflow Hub 层。这一层使用一个预训练的保存好的模型来将句子映射为嵌入向量（embedding vector）。我们所使用的预训练文本嵌入（embedding）模型(google/tf2-preview/gnews-swivel-20dim/1)将句子切割为符号, 嵌入（embed）每个符号然后进行合并。最终得到的维度是：(num_examples, embedding_dimension)。
# 该定长输出向量通过一个有 16 个隐层单元的全连接层（Dense）进行管道传输。
# 最后一层与单个输出结点紧密相连。使用 Sigmoid 激活函数, 其函数值为介于 0 与 1 之间的浮点数, 表示概率或置信水平。

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(train_data.shuffle(10000).batch(512), epochs=20, validation_data=validation_data.batch(512), verbose=1)

results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))
