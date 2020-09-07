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

# 从数据集中随机取出指定条数
train_data = train_data.sample(n=5000)

# 查看csv数据集信息
print(train_data.info())

# 定义一个词索引库
vocab = []

# 获取所有的单个词
for item in train_data.review:
    vocab.extend(item)

print(len(vocab))

# 去除重复的词
vocab = list(set(vocab))

print(len(vocab))

# 定义训练的X
_X_Data = []

# 构建X，得到一句话对应词库的索引
for review in train_data.review:
    arr = []
    for item in review:
        arr.append(vocab.index(item) + 1)
    _X_Data.append(arr)

print(_X_Data)

# 补齐数据长度操作
indexArr = []
for item in train_data.review:
     indexArr.append(len(item))

# 获取最长的句子长度
maxIndexLength = max(indexArr)

print(maxIndexLength)

# 补齐操作
for i in range(len(_X_Data)):
    _X_Data[i] = _X_Data[i] + [0] * (maxIndexLength - len(_X_Data[i]))
    _X_Data[i] = np.array(_X_Data[i])

_X_Data = np.array(_X_Data)

# 得到Y
_Y_Data = train_data.label


# 切分数据
train_X = _X_Data[:4000]

test_X = _X_Data[4000:]

train_Y = _Y_Data[:4000]

test_Y = _Y_Data[4000:]

print(len(vocab)+1)

# 创建模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=len(vocab)+1,output_dim=128,input_length=maxIndexLength))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64,activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(2,activation="softmax"))

model.compile(optimizer=tf.optimizers.Adam(),loss="sparse_categorical_crossentropy",metrics=["acc"])

model.fit(train_X,train_Y,epochs=5)

model.evaluate(test_X,test_Y)

_predict = model.predict(test_X)

# print(np.argmax(_predict[950]))

# _output = ""

# for item in test_X[950]:
#     if item == 0 : continue
#     _output += vocab[(item - 1)]

# print(_output)

print("*********************句子分类**********************")

_indices = np.random.permutation(1000)[:10]

for index in _indices:
    _argmax = np.argmax(_predict[index])
    _output = ""
    for item in test_X[index]:
        if item == 0 : continue
        _output += vocab[(item - 1)]
    print("预测的标签为：{0},真实的标签为：{1},句子：{2}".format(_argmax,test_Y.values[index],_output))



print("*********************语义提取**********************")

def get_bert_encode(inputs):
    embedding_layer = model.layers[0]
    pooling = tf.keras.layers.Flatten()
    encodes = []

    for i in tqdm(range(0,len(inputs),32)):
        batch = inputs[i:i+32]
        pooled = pooling(embedding_layer(batch))        
        encodes.extend(pooled.numpy())
        # pass
    return np.array(encodes)


# 获取模型中的embedding层
embedding_layer = model.layers[0]


print(get_bert_encode(test_X[0]))

# 根据词数据获取词向量
x_word_embedding = np.array(embedding_layer(test_X[0]))

train_X_word_embedding = np.array(embedding_layer(train_X))

scores = scipy.spatial.distance.cdist(x_word_embedding, train_X_word_embedding)[0]
results = zip(range(len(scores)), scores)
results = sorted(results, key=lambda x: x[1], reverse=False) 
for idx, score in results[0:20]:
        print(train_data.review.values[idx], "(Score: %.4f)" % (score))
