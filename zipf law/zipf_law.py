import nltk

nltk.download('brown')
from nltk.corpus import brown
from collections import Counter
import string

punc = set(string.punctuation)
words = [w.lower() for w in brown.words() if w not in punc]
counts = Counter(words)
print(counts.most_common(10))