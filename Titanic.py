
#
# 泰坦尼克号生还者预测
#

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

encoder = LabelEncoder()

train = pd.read_csv("D:\\Google下载\\titanic\\train.csv")

evel = pd.read_csv("D:\\Google下载\\titanic\\test.csv")

print("train：{0}".format(len(train)))

print("evel：{0}".format(len(evel)))


train_Y = train.Survived

train['Fare'] = train['Fare'].fillna(train['Fare'].mean())

train['Age'] = train['Age'].fillna(train['Age'].mean())

train['Embarked'] = train['Embarked'].fillna( 'S' )
#缺失数据比较多，船舱号（Cabin）缺失值填充为U，表示未知（Uknow） 
train['Cabin'] = train['Cabin'].fillna( 'U' )

print(train.info())

for colunm in train:
    if colunm in ["Name","Sex","Ticket","Cabin","Embarked"]:
        # if colunm == "Cabin" or colunm == "Embarked" :
        #     train[colunm] = map(str,train[colunm])
        train[colunm] = encoder.fit_transform(train[colunm])

train_X = train.iloc[:,2:12]

print(train_X)
print(train_Y)


evel['Fare'] = evel['Fare'].fillna(evel['Fare'].mean())

evel['Age'] = evel['Age'].fillna(evel['Age'].mean())

evel['Embarked'] = evel['Embarked'].fillna( 'S' )
#缺失数据比较多，船舱号（Cabin）缺失值填充为U，表示未知（Uknow） 
evel['Cabin'] = evel['Cabin'].fillna( 'U' )

for colunm in evel:
    if colunm in ["Name","Sex","Ticket","Cabin","Embarked"]:
        # if colunm == "Cabin" or colunm == "Embarked" :
        #     evel[colunm] = map(str,evel[colunm])
        evel[colunm] = encoder.fit_transform(evel[colunm])

evel_X = evel.iloc[:,1:11]


# print(train_X.shape)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128,input_shape=(None,10)))
model.add(tf.keras.layers.Dense(64,activation="relu"))
model.add(tf.keras.layers.Dense(32,activation="relu"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

model.compile(tf.keras.optimizers.Adam(learning_rate=0.0001),loss="mse",metrics=["accuracy"])

modelInfo = model.fit(train_X,train_Y,epochs=500)

plt.plot(modelInfo.epoch,modelInfo.history.get("loss"))
plt.show()

#model.evaluate(evel_X,evel_Y,batch_size=264)

_predict = model.predict(evel_X)

print(_predict)

print(np.round(_predict))
