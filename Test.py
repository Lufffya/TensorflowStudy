import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


dataSet = pd.read_csv("Test.csv")

print(dataSet)


X = dataSet.X

print(X)


Y = dataSet.Y

print(Y)

#plt.scatter(X,Y)
#plt.show()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1,input_shape=(1,)))

model.compile(optimizer="adam",loss="mse")

print(model.summary())

model.fit(X,Y,epochs=5000)


print(model.predict([8]))