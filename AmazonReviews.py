import bert
import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import scipy.spatial
from tqdm import tqdm
from sklearn.cluster import KMeans
from matplotlib import pyplot


'''读取模型'''

bert_params = bert.params_from_pretrained_ckpt("Models\\Pretraining_Bert_EN_Uncased_L-12_H-768_A-12_Google-Research")

bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert")

tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file="Models\\Bert_EN_Uncased_L-12_H-768_A-12_Google-Research\\vocab.txt")


'''语义提取'''
max_seq_length = 128

train_data = pd.read_csv("DataSet\AmazonReviews(B07PBV7D48).csv")

print(len(train_data))

train_X = []
for review in train_data.values:
    tokens = tokenizer.tokenize(list(review)[0])
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    if len(token_ids) < max_seq_length:
        token_ids = token_ids + [tokenizer.vocab["[unused1]"]] * (max_seq_length - len(token_ids))
    token_ids = token_ids[:max_seq_length]
    train_X.append(np.array(token_ids))
train_X = np.array(train_X)


def Get_Encoding(inputs):
    outPut = []
    pooling = tf.keras.layers.GlobalAveragePooling1D()
    for i in tqdm(range(0,len(inputs),32)):
        batch = inputs[i:i+32]
        batch_word_embedding = pooling(bert_layer(batch))
        outPut.extend(batch_word_embedding.numpy())
    return np.array(outPut)

# 解析标签
target = ["very easy to use"]
target_Token = tokenizer.tokenize(target[0])
target_ids = tokenizer.convert_tokens_to_ids(target_Token)
if len(target_ids) < max_seq_length:
    target_ids = target_ids + [tokenizer.vocab["[unused1]"]] * (max_seq_length - len(target_ids))
    target_ids = target_ids[:max_seq_length]
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
    print(train_data.values[_index], "(Distance: %.4f)" % (distance))



'''文本聚类'''
clusterData =  train_X_Encoding

# 初始化聚类函数
kMeans_Model = KMeans(n_clusters=20,max_iter=500,tol=0.0001,verbose=True,n_jobs=10)

# 训练或者叫做执行聚类操作
kMeans_Model.fit(clusterData)

# K组数据点的每个中心点
centers = kMeans_Model.cluster_centers_

# 每个数据点所属分组
labels = kMeans_Model.labels_

print(list(set(labels)))

firstClassIndex = []

for index,item in enumerate(labels):
    if item == 0 :
        firstClassIndex.append(index)


firstClassResult = []

for index in firstClassIndex:
    firstClassResult.append(train_data.values[index])


import csv

csv_File = open('DataSet\新建文本文档.csv','w',encoding='utf-8')

csv_writer = csv.writer(csv_File)

for review in firstClassResult:
    csv_writer.writerow(review)

r1 = pd.Series(kMeans_Model.labels_).value_counts() #统计各个类别的数目


r2 = pd.DataFrame(kMeans_Model.cluster_centers_) #找出聚类中心
 



# for i in range(len(labels)):
#     pyplot.scatter(x[i][0], x[i][1], c=('r' if labels[i] == 0 else 'b'))
# pyplot.scatter(centers[:,0],centers[:,1],marker='*', s=100)
 

# # 预测
# predict = target_X_Encoding

# label = clf.predict(predict)
# for i in range(len(label)):
#     pyplot.scatter(predict[i][0], predict[i][1], c=('r' if label[i] == 0 else 'b'), marker='x')
 
# pyplot.show()



print()