
#
# Google Bert 模型
#
import tensorflow as tf
import numpy as np
import bert # pip install bert-for-tf2
import os
import pandas as pd

# 从Bert开源项目中下载的bert中文模型
# https://github.com/google-research/bert

# 获取bert模型预训练的参数
bert_params = bert.params_from_pretrained_ckpt("Models\Bert_Chinese")

# 从参数构建bert模型层
bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert")

# 从bert词库构建标记器
tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file="Models\Bert_Chinese\\vocab.txt")

# 读取数据集
train_data = pd.read_csv("DataSet\外卖评价.csv")

# 从数据集中随机取出指定条数
train_data = train_data.sample(n=11000)

train_Y = train_data.label.values

train_X = []

# 获取最大词长度
_reviewsLenth = []
for reviews in train_data.review.values:
    _reviewsLenth.append(len(reviews))

# maxReviewCount = max(_reviewsLenth)
maxReviewCount = 256

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


model.fit(train_X,train_Y,epochs=2)
