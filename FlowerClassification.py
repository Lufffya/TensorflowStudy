


#本教程遵循基本的机器学习工作流程：

# 1,检查并了解数据
# 2,建立输入管道
# 3,建立模型
# 4,训练模型
# 5,测试模型
# 6,改进模型并重复该过程

#
# 花朵照片分类
# https://tensorflow.google.cn/tutorials/images/classification
#

import pathlib
import tensorflow as tf


dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)


train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir,validation_split=0.2,subset="training",seed=123,image_size=(180, 180),batch_size=32)


val_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir,validation_split=0.2,subset="validation",seed=123,image_size=(180, 180),batch_size=32)


# model = tf.keras.Sequential([
#   tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(180, 180, 3)),

#   tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(5)
# ])


_input = tf.keras.Input(shape=(180, 180, 3))

layer1 = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

layer2 = layer1(_input)

layer3 = tf.keras.layers.Conv2D(16,3,padding="same")

layer4 = layer3(layer2)

layer5 = tf.keras.activations.relu(layer4)

layer6 = tf.keras.layers.MaxPool2D()

layer7 = layer6(layer5)

layer8 = tf.keras.layers.Flatten()

layer9 = layer8(layer7)

layer10 = tf.keras.layers.Dense(128)

layer11 = layer10(layer9)

layer12 = tf.keras.activations.relu(layer11)

layer13 = tf.keras.layers.Dense(5)

_output = layer13(layer12)

model = tf.keras.Model(inputs=_input,outputs=_output)

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])


print(model.summary())


history = model.fit(train_ds,validation_data=val_ds,epochs=10)