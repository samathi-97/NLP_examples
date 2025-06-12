import nltk
from nltk.tokenize import TreebankWordTokenizer
sentence = "Isn't Natural Language Processing just fascinating?"
tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(sentence)
normalized_tokens = [token.lower() for token in tokens]
print(normalized_tokens)
