#
# 标注工具
#

import bert
import numpy as np
import tensorflow as tf
import xlrd

max_seq_length = 128
bert_params = bert.params_from_pretrained_ckpt("Models\\Pretraining_Bert_EN_Uncased_L-12_H-768_A-12_Google-Research")
bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert")
tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file="Models\\Bert_EN_Uncased_L-12_H-768_A-12_Google-Research\\vocab.txt")
workbook = xlrd.open_workbook("DataSet\\ReviewKMaensResult.xlsx")

train_Data = []
train_Label = []

for index,sheet in enumerate(workbook.sheets()):
    print(f"====={str(sheet.name)}=====")
    print("总行数：" + str(sheet.nrows))
    print("总列数：" + str(sheet.ncols))
    sheetReviews = sheet.col_values(0)
    train_Data.extend(sheetReviews)
    train_Label.extend([index]*sheet.nrows)
    print(len(train_Data))
    print(len(train_Label))

train_X = []
for reviews in train_Data:
    tokens = tokenizer.tokenize(reviews)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    if len(token_ids) < max_seq_length:
        token_ids = token_ids + [tokenizer.vocab["[unused1]"]] * (max_seq_length - len(token_ids))
    token_ids = token_ids[:max_seq_length]
    train_X.append(np.array(token_ids))

indices = np.random.permutation(len(train_Data))
train_X = np.array(train_X)[indices]
train_Label = np.array(train_Label)[indices]

model = tf.keras.Sequential()
model.add(bert_layer)
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(1024,activation="relu"))
model.add(tf.keras.layers.Dense(512,activation="relu"))
model.add(tf.keras.layers.Dense(64,activation="relu"))
model.add(tf.keras.layers.Dense(20,activation="softmax"))

model.build(input_shape=(None,max_seq_length))

print(model.summary())

model.compile(optimizer=tf.optimizers.Adam(lr=0.00001),loss=tf.losses.SparseCategoricalCrossentropy(),metrics=["acc"])

model.fit(train_X,train_Label,epochs=2)