import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


print(tf.__version__)

X = [1,2,3,4,5,6,7,8,9,10]

Y = [10,20,30,40,50,60,70,80,90,100]


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1,input_shape=(1,)))
<<<<<<< HEAD
# model.add(tf.keras.layers.Dense(2,activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.03),loss="mse")

print(model.summary())

# model.fit(X,Y,batch_size=32,epochs=5000)

# print(model.predict([11]))
=======

model.compile(optimizer="adam",loss="mse")

print(model.summary())

model.fit(X,Y,epochs=500)

print(model.predict([11]))



# fashion_mnist = tf.keras.datasets.fashion_mnist

# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
>>>>>>> d4a276319d6691325557bd76e6e5974ea008dfda


# dataSet = pd.read_csv("Test.csv")

# print(dataSet)


# X = dataSet.X

# print(X)


# Y = dataSet.Y

# print(Y)

#plt.scatter(X,Y)
#plt.show()