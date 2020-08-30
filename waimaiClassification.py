#
# 外卖好评差评分类
#

import tensorflow as tf
import pandas as pd


train_data = pd.read_csv("外卖评价.csv")

# 从数据集中随机取出指定条数
train_data = train_data.sample(n=5000)

print(train_data.info())

# 定义一个词索引库
vocab = []

for item in train_data.review:
    vocab.extend(item)

print(len(vocab))

vocab = set(vocab)

print(len(vocab))

train_X = []


