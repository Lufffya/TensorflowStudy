import numpy as np
import tensorflow as tf

print(np.array([[1,2,3,4],[1,2,3,4]]).shape)


x1_input= np.arange(10).reshape(5, 2)

print(x1_input.shape)

x1 = tf.keras.layers.Dense(8)(x1_input)


x2_input = np.arange(10, 20).reshape(5, 2)

print(x2_input.shape)

x2 = tf.keras.layers.Dense(8)(x2_input)


concatted = tf.keras.layers.Concatenate()([x1, x2])

concatted.shape