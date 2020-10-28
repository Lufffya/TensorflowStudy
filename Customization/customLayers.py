#
# 自定义的网络层
# https://tensorflow.google.cn/tutorials/customization/custom_layers

import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 在tf.keras.层包，层是对象。建造一个层
# 只需构造对象。大多数层将数字作为第一个参数
# 输出尺寸/通道
layer = tf.keras.layers.Dense(100)

# 输入维度的数量通常是不必要的，因为它可以推断出来
# 第一次使用该层时，如果需要，可以提供该层
# 手动指定它，这在某些复杂模型中很有用
layer = tf.keras.layers.Dense(10, input_shape=(None, 5))

# 要使用层，只需调用它
print(tf.zeros([10, 5]))
print(layer(tf.zeros([10, 5])))


# 实现自定义层
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel", shape=[int(input_shape[-1]), self.num_outputs])

    def call(self, input):
        return tf.matmul(input, self.kernel)


layer = MyDenseLayer(10)


print(layer(tf.zeros([10, 5])))

print([var.name for var in layer.trainable_variables])


class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(
            filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)


block = ResnetIdentityBlock(1, [1, 2, 3])

print(block(tf.zeros([1, 2, 3, 3])))

block.summary()


# 但是，在许多情况下，组成多层的模型只是简单地将一层称为另一层。使用tf.keras.Sequential以下代码可以做到这一点：
my_seq = tf.keras.Sequential([
    tf.keras.layers.Conv2D(1, (1, 1), input_shape=(None, None, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(2, 1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(3, (1, 1)),
    tf.keras.layers.BatchNormalization()])

print(my_seq(tf.zeros([1, 2, 3, 3])))
my_seq.summary()
