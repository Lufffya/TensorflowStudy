

import tensorflow as tf
import numpy as np


fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


train_images = train_images / 255.0

test_images = test_images / 255.0

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128,activation="relu"))
model.add(tf.keras.layers.Dense(64,activation="relu"))
model.add(tf.keras.layers.Dense(32,activation="relu"))
model.add(tf.keras.layers.Dense(10,activation="softmax"))


model.compile(optimizer="adam",loss="sparse_categorical_crossentropy")


model.fit(train_images,train_labels,epochs=20)


predictions = model.predict(test_images)


print(predictions[0])



print(np.argmax(predictions[0]))

print(test_labels[0])