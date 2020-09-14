
#
# Google Bert 模型
#
import tensorflow as tf
import numpy as np
import bert # pip install bert-for-tf2
import os
import pandas as pd
import scipy.spatial
from tqdm import tqdm

# 从Bert开源项目中下载的bert中文模型
# https://github.com/google-research/bert

# 获取bert模型预训练的参数
bert_params = bert.params_from_pretrained_ckpt("Models\Bert_CN_Google-Research")

# 从参数构建bert模型层
bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert")

# 从bert词库构建标记器
tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file="Models\Bert_CN_Google-Research\\vocab.txt")

# 读取数据集
train_data = pd.read_csv("DataSet\外卖评价.csv")

# 从数据集中随机取出指定条数
train_data = train_data.sample(n=10000)

train_Y = train_data.label.values

train_X = []

# 获取最大词长度
_reviewsLenth = []
for reviews in train_data.review.values:
    _reviewsLenth.append(len(reviews))

# maxReviewCount = max(_reviewsLenth)
maxReviewCount = 128

# 对中文进行编码
for reviews in train_data.review.values:
    tokens = tokenizer.tokenize(reviews)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    if len(token_ids) < maxReviewCount:
        token_ids = token_ids + [tokenizer.vocab["[unused1]"]] * (maxReviewCount - len(token_ids))
    token_ids = token_ids[:maxReviewCount]
    train_X.append(np.array(token_ids))

train_X = np.array(train_X)

model = tf.keras.Sequential()
model.add(bert_layer)
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(1024,activation="relu"))
model.add(tf.keras.layers.Dense(512,activation="relu"))
model.add(tf.keras.layers.Dense(64,activation="relu"))
model.add(tf.keras.layers.Dense(2,activation="softmax"))

model.build(input_shape=(None,maxReviewCount))

print(model.summary())

model.compile(optimizer=tf.optimizers.Adam(lr=0.00001),loss=tf.losses.SparseCategoricalCrossentropy(),metrics=["acc"])

model.fit(train_X,train_Y,epochs=2,batch_size=32)

def Get_Encoding(inputs):
    outPut = []
    # bert_layer = model.layers[0]
    pooling = tf.keras.layers.GlobalAveragePooling1D()
    for i in tqdm(range(0,len(inputs),32)):
        batch = inputs[i:i+32]
        batch_word_embedding = pooling(bert_layer(batch))
        outPut.extend(batch_word_embedding.numpy())
    return np.array(outPut)

# 解析标签
target = ["味道好，送餐速度也快"]
target_Token = tokenizer.tokenize(target[0])
target_ids = tokenizer.convert_tokens_to_ids(target_Token)
if len(target_ids) < maxReviewCount:
    target_ids = target_ids + [tokenizer.vocab["[unused1]"]] * (maxReviewCount - len(target_ids))
    target_ids = target_ids[:maxReviewCount]
target_ids = np.array([np.array(target_ids)])

# 获取标签句子的词向量
target_X_Encoding = Get_Encoding(target_ids)

# 获所有训练句子的词向量
train_X_Encoding = Get_Encoding(train_X)

# 计算标签句子词向量和训练句子词向量之间的欧式距离
euclidean_Distance = scipy.spatial.distance.cdist(target_X_Encoding, train_X_Encoding)[0]

_zip = zip(range(len(euclidean_Distance)),euclidean_Distance)

_sorted = sorted(_zip,key=lambda x: x[1],reverse=False)

for _index, distance in _sorted[0:20]:
    print(train_data.review.values[_index], "(Distance: %.4f)" % (distance))

print()