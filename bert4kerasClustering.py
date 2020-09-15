import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
import xlwt
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer

config_path = 'Models\\Pretraining_Bert_EN_Uncased_L-12_H-768_A-12_Google-Research\\bert_config.json'
checkpoint_path = 'Models\\Pretraining_Bert_EN_Uncased_L-12_H-768_A-12_Google-Research\\model.ckpt-94000'
dict_path = 'Models\\Bert_EN_Uncased_L-12_H-768_A-12_Google-Research\\vocab.txt'
tokenizer = Tokenizer(dict_path, do_lower_case=True)
model = build_transformer_model(config_path, checkpoint_path)

max_seq_length = 32

# 过滤句子长度
train_data = []
for item in pd.read_csv("DataSet\\AmazonReviews(B07PBV7D48).csv").values :
    token_ids,segment_ids = tokenizer.encode(item[0])
    if len(token_ids) > max_seq_length+2:
        continue 
    train_data.append(item)
train_data = np.array(train_data)

# 提句向量特征
train_Data_Embedding = []
for index, review in tqdm(enumerate(train_data)):
    token_ids, segment_ids = tokenizer.encode(review[0], maxlen=max_seq_length+2)
    newTensor = []
    word_Embedding_Tensor = model.predict([np.array([token_ids]), np.array([segment_ids])])[0]
    for i in range(768):
        temp = 0
        for j in range(len(word_Embedding_Tensor)): 
            temp += word_Embedding_Tensor[j][i]
        newTensor.append(temp/(len(word_Embedding_Tensor)))
    train_Data_Embedding.append(np.array(newTensor))
train_Data_Embedding = np.array(train_Data_Embedding)


# '''文本聚类'''
# 初始化聚类函数
kMeans_Model = KMeans(n_clusters=20,max_iter=5000,tol=0.00001,verbose=True,n_jobs=10)

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
    for _index,_item in enumerate(labels):
        if _item == _label :
            classIndex.append(_index)

    classResult = []
    for index in classIndex:
        classResult.append(train_data[index])

    sheet = workbook.add_sheet("聚类{0}".format(_label))
    for row in range(0, len(classResult)):
        sheet.write(row,0,str(classResult[row][0]))
    
workbook.save(r'DataSet\\test.xlsx')

print()