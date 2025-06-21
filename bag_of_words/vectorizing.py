import sklearn
from sklearn.feature_extraction.text import CountVectorizer

# Sample NLP documents
docs = [
    "Natural Language Processing helps machines understand human language.",
    "NLP is used in chatbots, translation apps, and voice assistants.",
    "Text classification is one of the key tasks in NLP."
]

# Create the vectorizer and fit it to the docs
vectorizer = CountVectorizer()
count_vectors = vectorizer.fit_transform(docs)

# Show the vectorized form
print(count_vectors.toarray())

# See the vocabulary (columns)
print(vectorizer.get_feature_names_out())
