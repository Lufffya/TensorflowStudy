import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()


plt.figure()
for i in range(100):
    plt.subplot(10, 10, i+1)
    plt.imshow(train_images[i], cmap="gray")
    plt.axis(False)
plt.show()


train_images = (train_images / 127.5) - 1
train_images = train_images.reshape(60000, 784)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation="tanh"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation="tanh"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation="softmax"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation="tanh"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(512, activation="tanh"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(784, activation="sigmoid"),
])

model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
model.fit(train_images, train_images, batch_size=128, epochs=20)

pre = model.predict(train_images[:100]).reshape((100, 28, 28))

# plt.figure()
# for i in range(100):
#     plt.subplot(10, 10, i+1)
#     plt.imshow(pre[i], cmap="gray")
#     plt.axis(False)
# plt.show()


_layer0 = model.get_layer(index=0)
_layer1 = model.get_layer(index=1)
_layer2 = model.get_layer(index=2)
_layer3 = model.get_layer(index=3)
_layer4 = model.get_layer(index=4)

g_model = tf.keras.Sequential([_layer0,_layer1,_layer2,_layer3,_layer4])

_labels = g_model(train_images[:100]).numpy()

plt.figure()
for i in range(len(_labels)):
    _lable = np.argmax(_labels[i])
    plt.subplot(10, 10, i+1)
    plt.title(_lable)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.imshow(pre[i], cmap="gray")
    plt.axis(False)
plt.show()

print()