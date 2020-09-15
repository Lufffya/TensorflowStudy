



import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot
 
# 要分类的数据点
x = np.array([ [1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11] ])
# pyplot.scatter(x[:,0], x[:,1])


# 把上面数据点分为两组（非监督学习）
clf = KMeans(n_clusters=2)
clf.fit(x)  # 分组
 
centers = clf.cluster_centers_ # 两组数据点的中心点
labels = clf.labels_   # 每个数据点所属分组
print(centers)
print(labels)
 
for i in range(len(labels)):
    pyplot.scatter(x[i][0], x[i][1], c=('r' if labels[i] == 0 else 'b'))
pyplot.scatter(centers[:,0],centers[:,1],marker='*', s=100)
 
# 预测
predict = [[2,1], [6,9]]
label = clf.predict(predict)
for i in range(len(label)):
    pyplot.scatter(predict[i][0], predict[i][1], c=('r' if label[i] == 0 else 'b'), marker='x')
 
pyplot.show()

