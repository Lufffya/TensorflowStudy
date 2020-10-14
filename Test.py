import xlrd

data = xlrd.open_workbook("D:/Google下载/US关键词黑名单汇总表.xlsx")

table = data.sheets()[0]

values = table.col_values(0)

with open("douban.txt","w",encoding='utf-8') as f:
        strr = ";".join(values)
        f.write(strr)


import pandas as pd
import stanza

nlp = stanza.Pipeline(dir="Models\\Stanford_CoreNLP\\Stanford_EN_Model")

review_Data = pd.read_csv("DataSet\\AmazonReviews.csv")


for review in review_Data:
    doc = nlp(review)
    print(f'======{review}=======')
    for ent in doc.ents:
        print(f'entity: {ent.text}\ttype: {ent.type}', sep='\n')