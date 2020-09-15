#
# 外卖好评差评分类
#

import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.spatial
from tqdm import tqdm

# 读取数据集
train_data = pd.read_csv("DataSet\外卖评价.csv")

take_OutData = 10000
train_Size = 8000
_random = take_OutData - train_Size

# 从数据集中随机取出指定条数
train_data = train_data.sample(n=take_OutData)

# 查看csv数据集信息
print(train_data.info())

# 定义一个词索引库
vocab = []

# 获取所有的单个词
for item in train_data.review:
    vocab.extend(item)

# print(len(vocab))

# 去除重复的词
vocab = list(set(vocab))

# print(len(vocab))

# 定义训练的X
_X_Data = []

# 构建X，得到一句话对应词库的索引
for review in train_data.review:
    arr = []
    for item in review:
        arr.append(vocab.index(item) + 1)
    _X_Data.append(arr)

# print(_X_Data)

# 补齐数据长度操作
indexArr = []
for item in train_data.review:
     indexArr.append(len(item))

# 获取最长的句子长度
maxIndexLength = max(indexArr)

# print(maxIndexLength)

# 补齐操作
for i in range(len(_X_Data)):
    _X_Data[i] = _X_Data[i] + [0] * (maxIndexLength - len(_X_Data[i]))
    _X_Data[i] = np.array(_X_Data[i])

_X_Data = np.array(_X_Data)

# 得到Y
_Y_Data = train_data.label

# 切分数据
train_X = _X_Data[:train_Size]

test_X = _X_Data[train_Size:]

train_Y = _Y_Data[:train_Size]

test_Y = _Y_Data[train_Size:]

# print(len(vocab)+1)

# 创建模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=len(vocab)+1,output_dim=256,input_length=maxIndexLength))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(128,activation="relu"))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(64,activation="relu"))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(2,activation="softmax"))

model.compile(optimizer=tf.optimizers.Adam(),loss="sparse_categorical_crossentropy",metrics=["acc"])

model.fit(train_X,train_Y,epochs=5)

model.evaluate(test_X,test_Y)

_predict = model.predict(test_X)


print("*********************句子分类**********************")

_indices = np.random.permutation(_random)[:10]
for index in _indices:
    _argmax = np.argmax(_predict[index])
    review = ""
    for item in test_X[index]:
        if item == 0 : continue
        review += vocab[(item - 1)]
    print("softmax概率值：{0}，预测的标签为：{1}，真实的标签为：{2}，词句：{3}".format(_predict[index],_argmax,test_Y.values[index],review))


print("*********************语义提取**********************")

# 根据索引获取词向量输出
def get_encode(inputs):
    # 获取训练模型中的embedding层
    embedding_layer = model.layers[0]
    # 定义Flatten层
    flatten = tf.keras.layers.GlobalAveragePooling1D()
    # 定义输出
    encodes = []
    # 从0开始，对inputs数据进行批处理，此时批处理大小为32
    for i in range(0,len(inputs),32):
        # 获取批处理大小的数据
        batch = inputs[i:i+32]
        # 传入embedding层进行处理，输出词向量，即词在embedding网络层的多维空间中的位置
        batch_word_embedding = flatten(embedding_layer(batch))
        # 每次记录
        encodes.extend(batch_word_embedding.numpy())
    return np.array(encodes)

# 随机从测试数据集中选取一条评论
_index = int(np.random.permutation(_random)[:1])
review = ""
for item in test_X[_index]:
    if item == 0 : continue
    review += vocab[(item - 1)]
_argmax = np.argmax(_predict[_index])
print("softmax概率值：{0}，预测的标签为：{1}，真实的标签为：{2}，词句：{3}".format(_predict[_index],_argmax,test_Y.values[_index],review))

# target = ["味道好，送餐速度也快"]

# target_X = []

# for item in target[0]:
#     target_X.append(vocab.index(item))
# if len(target_X) < maxIndexLength:
#     target_X =target_X + [0] * (maxIndexLength - len(target_X))

# print(target_X)

# target_X = np.array([target_X])

# 获取当前词句子数据
target_X = np.array([test_X[_index]])

# 获取当前测试句子的词向量
x_word_embedding = get_encode(target_X)

# 获取所有训练句子的词向量
train_X_word_embedding = get_encode(train_X)

# 该函数用于计算两个输入集合的距离,默认metric='euclidean'表示计算欧式距离
# 返回两元素之间的距离
euclidean_Distance = scipy.spatial.distance.cdist(x_word_embedding, train_X_word_embedding)[0]

# zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
results = zip(range(len(euclidean_Distance)), euclidean_Distance)

# sorted函数对所有可迭代的对象进行排序操作
# param1：可迭代对象：results
# param2：定可迭代对象中的一个元素来进行排序
# param3：reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）
results = sorted(results, key=lambda x: x[1], reverse=False)

# 上述对距离进行升序排序，此时取出距离最近的元素，也相当于取出和预测词句意思最相近的词句
for _index, distance in results[0:20]:
    review = ""
    for item in train_X[_index]:
        if item == 0 : continue
        review += vocab[(item - 1)]
    print("真实的标签为：{0}，词句：{1}".format(train_Y.values[_index],review),"(Distance: %.4f)" % (distance))