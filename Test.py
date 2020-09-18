

import stanza


nlp = stanza.Pipeline(lang='en', processors='tokenize',dir="Models\\Stanford_CoreNLP\\Stanford_EN_Model")

doc = nlp('This is a test sentence for stanza. This is another sentence.')

for i, sentence in enumerate(doc.sentences):

    print(f'====== Sentence {i+1} tokens =======')
    
    print(*[f'id: {token.id}\ttext: {token.text}' for token in sentence.tokens], sep='\n')