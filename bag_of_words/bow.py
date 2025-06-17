import spacy
spacy.cli.download("en_core_web_sm")
from collections import Counter
nlp = spacy.load("en_core_web_sm")
sentence = ("Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language.")
doc = nlp(sentence)

# Tokenize the sentence
tokens = [token.text for token in doc]
print(tokens)

# Count token frequencies
bag_of_words = Counter(tokens)
print(bag_of_words)


print(bag_of_words.most_common(5))