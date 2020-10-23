import tensorflow as tf
import numpy as np

x_train, x_test = np.array([1]), np.array([2])

print(x_train.shape)
print(x_test.shape)

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]


print(x_train.shape)
print(x_test.shape)
