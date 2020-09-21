


import pandas as pd
import stanza


nlp = stanza.Pipeline(dir="Models\\Stanford_CoreNLP\\Stanford_EN_Model")


review_Data = pd.read_csv("DataSet\\AmazonReviews.csv")


for review in review_Data:
    doc = nlp(review)
    print(f'======{review}=======')
    for ent in doc.ents:
        print(f'entity: {ent.text}\ttype: {ent.type}', sep='\n')