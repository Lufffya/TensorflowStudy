#
# 斯坦福大学NLP解决方案
#


# NLP Python包
import stanza

# 加载官方英文模型
nlp = stanza.Pipeline(dir="Models\\Stanford_CoreNLP\\Stanford_EN_Model")


#==========依存句法分析===========
doc = nlp('I Love you')

print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')


#==========实体标记提取===========
doc = nlp("Blue Light Blocking Glasses, 2Pack Cut UV400 Computer Reading Glasses for Anti Eyestrain")

print(*[f'entity: {ent.text}\ttype: {ent.type}' for ent in doc.ents], sep='\n')
print(*[f'token: {token.text}\tner: {token.ner}' for sent in doc.sentences for token in sent.tokens], sep='\n')


#==========情感分析===========
doc = nlp('I like it very much')

for i, sentence in enumerate(doc.sentences):
    print(i, sentence.sentiment)