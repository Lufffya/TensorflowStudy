
#
# 泰坦尼克号生还者预测
#

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()


train = pd.read_csv("Titanic_Train.csv")

evel = pd.read_csv("Titanic_Eval.csv")

print("train：{0}".format(len(train)))

print("evel：{0}".format(len(evel)))


train_Y = train.survived

for colunm in train:
    if colunm in ["sex","class","deck","embark_town","alone"]:
        train[colunm] = encoder.fit_transform(train[colunm])

train_X = train.iloc[:,1:10]


print(train_X)
print(train_Y)

evel_Y = evel.survived

for colunm in evel:
    if colunm in ["sex","class","deck","embark_town","alone"]:
        evel[colunm] = encoder.fit_transform(evel[colunm])

evel_X = evel.iloc[:,1:10]


print(train_X.shape)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128,input_shape=(None,9)))
model.add(tf.keras.layers.Dense(64,activation="relu"))
model.add(tf.keras.layers.Dense(32,activation="relu"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))


model.compile(optimizer="adam",loss="mse",metrics=["accuracy"])


model.fit(train_X,train_Y,validation_data=(evel_X,evel_Y),epochs=50)


model.evaluate(evel_X,evel_Y,batch_size=264)

# _predict = model.predict(evel_X)

# print(_predict[0])

