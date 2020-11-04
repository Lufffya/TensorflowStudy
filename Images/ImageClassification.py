#
# 花朵照片分类
# https://tensorflow.google.cn/tutorials/images/classification
#

# 本教程遵循基本的机器学习工作流程：
# 1,检查并了解数据
# 2,建立输入管道
# 3,建立模型
# 4,训练模型
# 5,测试模型
# 6,改进模型并重复该过程

import pathlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

'''准备数据集'''
data_dir = tf.keras.utils.get_file(
    'flower_photos', origin="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz", untar=True)
data_dir = pathlib.Path(data_dir)

# 创建数据集
batch_size = 32
img_height = 180
img_width = 180

# 读取训练数据集
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="training", seed=123, image_size=(img_height, img_width), batch_size=batch_size)

# 读取测试数据集
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="validation", seed=123, image_size=(img_height, img_width), batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

# 可视化数据
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# 配置数据集以提高性能
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 标准化数据
# 将[0, 1]使用“重缩放”图层将值标准化为该范围内的值
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

# 调用map将其应用于数据集
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

# 像素值现在以“[0,1]”为单位
print(np.min(first_image), np.max(first_image))


'''创建模型'''


class CNNModel(keras.models.Model):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.normalization = layers.experimental.preprocessing.Rescaling(
            1./255)
        self.conv2d1 = layers.Conv2D(16, 3, padding="same")
        self.conv2d1Pooling = layers.MaxPooling2D()
        self.conv2d2 = layers.Conv2D(32, 3, padding="same")
        self.conv2d2Pooling = layers.MaxPooling2D()
        self.conv2d3 = layers.Conv2D(64, 3, padding="same")
        self.conv2d3Pooling = layers.MaxPooling2D()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128)
        self.dense2 = layers.Dense(5)

    def call(self, inputs):
        x = self.normalization(inputs)
        x = self.conv2d1(x)
        x = keras.activations.relu(x)
        x = self.conv2d1Pooling(x)
        x = self.conv2d2(x)
        x = keras.activations.relu(x)
        x = self.conv2d2Pooling(x)
        x = self.conv2d3(x)
        x = keras.activations.relu(x)
        x = self.conv2d3Pooling(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = keras.activations.relu(x)
        return self.dense2(x)


model = CNNModel()
model.build(input_shape=(None, img_height, img_width, 3))
model.summary()

'''训练'''
epochs = 10

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

train_loss = tf.keras.metrics.Mean()
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

for epoch in range(epochs):
    for images, labels in train_ds:
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)
    # 打印结果
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(
        epoch, train_loss.result(), train_accuracy.result()))


# model.compile(optimizer=tf.keras.optimizers.Adam(
#     lr=0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])


# history = model.fit(train_ds, validation_data=val_ds, epochs=10)


# model.evaluate(val_ds)
