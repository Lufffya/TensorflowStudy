#
# 斯坦福大学NLP解决方案
#

# NLP Python包
import stanza

# 该stanza包需要GPU支持
# pip install https://download.pytorch.org/whl/cu101/torch-1.4.0-cp38-cp38-win_amd64.whl

# pip install https://download.pytorch.org/whl/cu101/torchvision-0.5.0-cp38-cp38-win_amd64.whl


# 加载官方英文模型
nlp = stanza.Pipeline(
    dir="Models\\Stanford_CoreNLP\\Stanford_EN_Model", processors="tokenize,sentiment")

#==========情感分析===========#
doc = nlp(
    'I love the feel of the mouse it is self but it broke within 4 months of having it')

for i, sentence in enumerate(doc.sentences):
    print(i, sentence.sentiment)

doc = nlp('did not like it')

for i, sentence in enumerate(doc.sentences):
    print(i, sentence.sentiment)

doc = nlp('I do not like it very much')

for i, sentence in enumerate(doc.sentences):
    print(i, sentence.sentiment)

doc = nlp('I changed the battery and it still does not work')

for i, sentence in enumerate(doc.sentences):
    print(i, sentence.sentiment)

# import xlrd

# # 打开文件
# review_Data = xlrd.open_workbook('DataSet\\ReviewKMaensResult.xlsx')

# # 查看工作表
# print("工作表：" + str(review_Data.sheet_names()))

# # 遍历所有Sheet
# for sheet in review_Data.sheets():
#     print("============================")
#     print("Sheet名称：" + str(sheet.name))
#     print("总行数：" + str(sheet.nrows))
#     print("总列数：" + str(sheet.ncols))
#     sheetReviews = sheet.col_values(0)

#     for review in sheetReviews:
#         doc = nlp(review)
#         print(f'======{review}=======')
#         for ent in doc.ents:
#             print(f'entity: {ent.text}\ttype: {ent.type}', sep='\n')

print()

#==========标记化和句子分割===========#
doc = nlp('This is a test sentence for stanza. This is another sentence.')

for i, sentence in enumerate(doc.sentences):
    print(f'====== Sentence {i+1} tokens =======')
    for token in sentence.tokens:
        print(f'id: {token.id}\ttext: {token.text}', sep='\n')


#==========依存句法分析===========#
doc = nlp('I Love you')

for sent in doc.sentences:
    for word in sent.words:
        print(
            f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}', sep='\n')


#==========实体标记提取===========#
doc = nlp("Blue Light Blocking Glasses, 2Pack Cut UV400 Computer Reading Glasses for Anti Eyestrain")

# 序列标注两种模式BIO和BIOES
# B，即Begin，表示开始
# I，即Intermediate，表示中间
# E，即End，表示结尾
# S，即Single，表示单个字符
# O，即Other，表示其他，用于标记无关字符

# BIO:
for ent in doc.ents:
    print(f'entity: {ent.text}\ttype: {ent.type}', sep='\n')

# BIOES:
for sent in doc.sentences:
    for token in sent.tokens:
        print(f'token: {token.text}\tner: {token.ner}', sep='\n')
