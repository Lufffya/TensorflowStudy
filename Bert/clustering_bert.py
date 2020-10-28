#
# 文本聚类
# 使用 bert4keras python包进行词向量提取
#

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
import xlwt
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer

# bert模型超参数配置文件
config_path = 'Models\\Pretraining_Bert_EN_Uncased_L-12_H-768_A-12_Google-Research\\bert_config.json'
# bert模型Tensorflow检查点文件,其中包含预先训练的权重
checkpoint_path = 'Models\\Pretraining_Bert_EN_Uncased_L-12_H-768_A-12_Google-Research\\model.ckpt-94000'
# bert模型词库,用于将词条映射到单词ID
dict_path = 'Models\\Bert_EN_Uncased_L-12_H-768_A-12_Google-Research\\vocab.txt'
# 获取bert词库标记器
tokenizer = Tokenizer(dict_path, do_lower_case=True)
# 构建bert模型
model = build_transformer_model(config_path, checkpoint_path)
# 定义最大输入词长
max_seq_length = 32

# 过滤原始数据中的词（根据词长）
train_data = []
for item in pd.read_csv("DataSet\\AmazonReviews(B07PBV7D48).csv").values:
    token_ids, segment_ids = tokenizer.encode(item[0])
    if len(token_ids) > max_seq_length+2:
        continue
    train_data.append(item)
train_data = np.array(train_data)

# 获取所有词句的句向量
# 句向量：表示词在多维空间中的位置,包含词的特征
train_Data_Embedding = []
for index, review in tqdm(enumerate(train_data)):
    # 获取词句对应的词库索引
    token_ids, segment_ids = tokenizer.encode(
        review[0], maxlen=max_seq_length+2)
    newTensor = []
    # 通过bert模型处理提取句向量
    word_Embedding_Tensor = model.predict(
        [np.array([token_ids]), np.array([segment_ids])])[0]
    # 对句向量进行处理
    # 因为逐一输入导致句子词向量长度不统一,所以无法进行接下来的聚类操作
    # 处理：循环句子的每个维度的句向量的总和除以句长度
    # 效果：对每个长度不一句子做句向量评价处理,以便保持相同的Shape
    for i in range(768):
        temp = 0
        for j in range(len(word_Embedding_Tensor)):
            temp += word_Embedding_Tensor[j][i]
        newTensor.append(temp/(len(word_Embedding_Tensor)))
    train_Data_Embedding.append(np.array(newTensor))
train_Data_Embedding = np.array(train_Data_Embedding)

# 初始化聚类函数
# n_clusters：聚类函数产生的类别数
# max_iter：最大迭代次数
# tol：允许的最小误差,若没到达迭代次数就满足该条件,则提前结束迭代
# verbose：显示执行过程输出信息
# n_jobs：开启10个线程执行
kMeans_Model = KMeans(n_clusters=20, max_iter=5000,
                      tol=0.00001, verbose=True, n_jobs=10)

# 训练或者叫做执行聚类操作
kMeans_Model.fit(train_Data_Embedding)

# K组数据点的每个中心点
centers = kMeans_Model.cluster_centers_

# 每个数据点所属分组
labels = kMeans_Model.labels_

print(list(set(labels)))

# 将聚类结果写入Execl
workbook = xlwt.Workbook(encoding='utf-8')

_class = list(set(labels))

for _label in _class:
    classIndex = []
    for _index, _item in enumerate(labels):
        if _item == _label:
            classIndex.append(_index)

    classResult = []
    for index in classIndex:
        classResult.append(train_data[index])

    sheet = workbook.add_sheet("聚类{0}".format(_label))
    for row in range(0, len(classResult)):
        sheet.write(row, 0, str(classResult[row][0]))

workbook.save(r'DataSet\\kMeansResult.xlsx')

print()
