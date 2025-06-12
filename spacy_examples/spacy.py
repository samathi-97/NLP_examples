import spacy

#Download an English language model
spacy.cli.download('en_core_web_sm')

nlp = spacy.load("en_core_web_sm")
doc = nlp("OpenAI is developing advanced NLP models in San Francisco with billions in funding.")

#Tokenization
for token in doc:
    print(token.text)

#Part-of-Speech Tagging (POS)
for token in doc:
    print(token.text, token.pos_)

#Named Entity Recognition (NER)
for ent in doc.ents:
    print(ent.text, ent.label_)

#Dependency Parsing
for token in doc:
    print(token.text, token.dep_, token.head.text)

#Lemmatization
for token in doc:
    print(token.text, "→", token.lemma_)
