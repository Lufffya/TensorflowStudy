#
# RGB 图像分类
#

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# 定义标签
lable_conf = {
    "buildings": {"cn": "建筑", "lable": 0},
    "forest": {"cn": "森林", "lable": 1},
    "glacier": {"cn": "冰川", "lable": 2},
    "mountain": {"cn": "山", "lable": 3},
    "sea": {"cn": "海", "lable": 4},
    "street": {"cn": "街道", "lable": 5}
}

# 一、构建训练集数据和标签
trainFilelDirectory = "E:\\PicturesClassification\\seg_train\\"

# 六分类，每个分类选取2000条训练数据
train_image = []

train_lable = []

for item in os.listdir(trainFilelDirectory):
    thisLable = lable_conf[item]["lable"]
    imageMaxCount = 2000
    for fileName in os.listdir(trainFilelDirectory + item):
        if imageMaxCount == 0:
            break
        img = cv2.imread(trainFilelDirectory + item +
                         "\\" + fileName)[:, :, ::-1]
        if img.shape != (150, 150, 3):
            continue
        # img = img / 255.
        train_image.append(img)
        train_lable.append(thisLable)
        imageMaxCount = imageMaxCount - 1
    print(len(train_image))

# 同时打乱训练数据和标签的顺序
# shape[0]表示第0轴的长度，通常是训练数据的数量
indices = np.random.permutation(2000 * 6)
train_image = np.array(train_image)[indices]
train_lable = np.array(train_lable)[indices]

# 二、构建测试集数据和标签
testFilelDirectory = "E:\\PicturesClassification\\seg_test\\"

# 六分类，每个分类选取400条测试数据
test_image = []

test_lable = []

for item in os.listdir(trainFilelDirectory):
    thisLable = lable_conf[item]["lable"]
    imageMaxCount = 400
    for fileName in os.listdir(trainFilelDirectory + item):
        if imageMaxCount == 0:
            break
        img = cv2.imread(trainFilelDirectory + item +
                         "\\" + fileName)[:, :, ::-1]
        if img.shape != (150, 150, 3):
            continue
        # img = img / 255.
        test_image.append(img)
        test_lable.append(thisLable)
        imageMaxCount = imageMaxCount - 1
    print(len(test_image))

# 同时打乱训练数据和标签的顺序
# shape[0]表示第0轴的长度，通常是训练数据的数量
indices = np.random.permutation(400 * 6)
test_image = np.array(test_image)[indices]
test_lable = np.array(test_lable)[indices]


model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(
    32, (3, 3), activation="relu", input_shape=(150, 150, 3)))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(6, activation="softmax"))

print(model.summary())

model.compile(optimizer=tf.optimizers.Adam(
), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["acc"])


model.fit(train_image, train_lable, epochs=10)

model.evaluate(test_image, test_lable)

_predict = model.predict(test_image)

# print(np.argmax(_predict[0]))

_indices = np.random.permutation(400 * 6)[:10]

# 输出预测的这个图像
plt.figure()
for i, index in enumerate(_indices):
    _title = ""
    _lable = np.argmax(_predict[index])
    if _lable == 0:
        _title = "建筑"
    elif _lable == 1:
        _title = "森林"
    elif _lable == 2:
        _title = "冰川"
    elif _lable == 3:
        _title = "山"
    elif _lable == 4:
        _title = "海"
    elif _lable == 5:
        _title = "街道"
    plt.subplot(2, 5, i+1)
    plt.title(_title)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.imshow(test_image[index])
plt.show()

print()
