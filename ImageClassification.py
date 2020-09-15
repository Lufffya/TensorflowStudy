#
# 图像分类
# 训练一个神经网络模型来对运动鞋和衬衫等衣物的图像进行分类
# https://tensorflow.google.cn/tutorials/keras/classification

# 引入包
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 引入mnist数据集
# mnist ：MNIST数据集(Mixed National Institute of Standards and Technology database)
# 是美国国家标准与技术研究院收集整理的大型手写数字数据库,包含60,000个示例的训练集以及10,000个示例的测试集
fashion_mnist = tf.keras.datasets.fashion_mnist

# 引入图像数据集
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 观察训练数据集图片形状以及对应标签分类
print(train_images.shape)
print(train_labels)

# 将像素值缩放到0至1的范围
train_images = train_images / 255.0
test_images = test_images / 255.0

# 建立一个神经网络序列模型
model = tf.keras.Sequential()
# 把多维的数据压至一维并告诉模型输入的数据集的形状，这里把输入是一个28*28像素二维数组压成 28 * 28 = 784 一维数组并使它排列成一排
model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) # 默认隐含层输出参数是 784
# 添加线性网络层，并定义每一层的输出
# 激活函数 relu 将保留线性有效部分
model.add(tf.keras.layers.Dense(512,activation="relu"))
model.add(tf.keras.layers.Dense(256,activation="relu"))
model.add(tf.keras.layers.Dense(128,activation="relu"))
model.add(tf.keras.layers.Dense(64,activation="relu"))
model.add(tf.keras.layers.Dense(32,activation="relu"))
# 最后使用softmax 多分类的激活函数输入10 个结果
model.add(tf.keras.layers.Dense(1))

print(model.summary())

# optimizer 优化函数 这里采用梯度下降优化算法
# loss 损失函数 sparse_categorical_crossentropy
# 计算标签和预测之间的交叉熵损失。
# 有两个或多个标签类别时，请使用此交叉熵损失函数。
# 我们希望标签以整数形式提供。如果要使用one-hot表示形式提供标签，请使用CategoricalCrossentropy损失
# metrics 指标显示 准确率（accuracy）
model.compile(optimizer="adam",loss="mse")

# 将训练数据集以及对应的标签添加到模型中训练
# epochs 训练的次数
model.fit(train_images,train_labels,epochs=5)

# predict 模型做出预测
# 在这里是给定一个图像,让模型预测该图像属于哪个分类
predict_images = model.predict(test_images[:1])

# 取出预测可信度的最大值
print(predict_images)

# 同时输入图像对应的真实标签 与预测的标签对比
print(test_labels[0])

# 输出预测的这个图像
# plt.figure()
# plt.imshow(test_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()
