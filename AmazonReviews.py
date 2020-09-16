#
# 亚马逊评论数据文本聚类
#

import xlwt
import bert
import numpy as np
import pandas as pd
import scipy.spatial
from tqdm import tqdm
import tensorflow as tf
from sklearn.cluster import KMeans


#加载Bert模型
bert_params = bert.params_from_pretrained_ckpt("Models\\Pretraining_Bert_EN_Uncased_L-12_H-768_A-12_Google-Research")
bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert")
tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file="Models\\Bert_EN_Uncased_L-12_H-768_A-12_Google-Research\\vocab.txt")
bert_Model = tf.keras.Sequential([bert_layer,tf.keras.layers.GlobalAveragePooling1D()])

# 获取所有句子的词索引
# max_seq_length = 128
train_data = pd.read_csv("DataSet\AmazonReviews(B07PBV7D48).csv")
print(len(train_data))
train_X = []
for review in train_data.values:
    tokens = tokenizer.tokenize(review[0])
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    if len(token_ids) > 32 : 
        continue
    train_X.append(np.array(token_ids))
train_X = np.array(train_X)
print(len(train_X))

# 根据句索引获取句向量
def Get_Encoding(inputs):
    outPut = []
    for item in tqdm(inputs):
        word_embedding = bert_Model.predict(item)
        # 对每个维度的每个句子长度做平均处理
        itemTensor  = []
        for i in range(768):
            tensor = 0
            for j in range(len(item)):
                tensor += word_embedding[j][i]
            itemTensor.append(tensor/(len(item)))      
        outPut.append(np.array(itemTensor))
    return np.array(outPut)

# 获所有训练句子的词向量
train_X_Encoding = Get_Encoding(train_X)

# 保存词向量到Execl
# workbook = xlwt.Workbook(encoding='utf-8')
# sheet = workbook.add_sheet("句向量")
# for row in range(len(train_X_Encoding)):
#     sheet.write(row,0,str(train_X_Encoding[row]))
# workbook.save(r'DataSet\\ReviewTensor.xlsx')

# 计算标签句子词向量和训练句子词向量之间的欧式距离
# euclidean_Distance = scipy.spatial.distance.cdist(target_X_Encoding, train_X_Encoding)[0]

# _zip = zip(range(len(euclidean_Distance)),euclidean_Distance)

# _sorted = sorted(_zip,key=lambda x: x[1],reverse=False)

# for _index, distance in _sorted[0:10]:
#     print(train_data.values[_index], "(Distance: %.4f)" % (distance))


# '''文本聚类'''
clusterData =  train_X_Encoding

# 初始化聚类函数
kMeans_Model = KMeans(n_clusters=20,max_iter=5000,tol=0.00001,verbose=True,n_jobs=10)

# 训练或者叫做执行聚类操作
kMeans_Model.fit(clusterData)

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
        classResult.append(train_data.values[index])

    sheet = workbook.add_sheet("聚类{0}".format(_label))
    for row in range(0, len(classResult)):
        sheet.write(row,0,str(classResult[row][0]))
    
workbook.save(r'DataSet\\test.xlsx')

#r1 = pd.Series(kMeans_Model.labels_).value_counts() #统计各个类别的数目
#r2 = pd.DataFrame(kMeans_Model.cluster_centers_) #找出聚类中心

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