
#
# 单元线性回归神经网络
#

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 定义X
X = [1,2,3,4,5,6,7,8,9,10]

# 定义Y
Y = [10,20,30,40,50,60,70,80,90,100]

# 创建模型
# 该模型拥有一个输入和一个输出
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1))
x = tf.keras.layers.Input((None,1))
layer1 = tf.keras.layers.Dense()
output1 = layer1(x)
model = tf.keras.Model(inputs=[x],outputs=[output1])
# model.add(tf.keras.layers.Dense(2,activation='softmax'))
model.build(input_shape=(None,1))
# 编译模型
# optimizer 优化器  Adam = 使用梯度下降优化算法  lr = 学习速率
# loss 损失函数 mse = 使用均方差
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.03),loss="mse")

print(model.summary())

# 训练模型
model.fit(X,Y,epochs=5000)

# 做出预测
print(model.predict([11]))

#plt.scatter(X,Y)
#plt.show()