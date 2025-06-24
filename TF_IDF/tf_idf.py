import math
from collections import Counter

corpus = [
    "Machine learning is amazing",
    "Deep learning is a branch of machine learning",
    "I love learning about AI and machine learning"
]
tokenized_corpus = [doc.lower().split() for doc in corpus]
total_docs = len(tokenized_corpus)


def compute_tf(doc_tokens):
    term_count = Counter(doc_tokens)
    total_terms = len(doc_tokens)
    tf = {term: count / total_terms for term, count in term_count.items()}
    return tf


tf_scores = [compute_tf(doc) for doc in tokenized_corpus]


def compute_df(corpus_tokens):
    df = {}
    for doc in corpus_tokens:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    return df


df = compute_df(tokenized_corpus)

# Step 4: Calculate IDF
idf = {
    term: math.log(total_docs / df[term])
    for term in df
}

# Step 5: Compute TF-IDF
tfidf_scores = []
for tf in tf_scores:
    tfidf = {term: tf[term] * idf[term] for term in tf}
    tfidf_scores.append(tfidf)

# 🔍 Example: Print TF-IDF for each document
for i, doc_tfidf in enumerate(tfidf_scores, 1):
    print(f"\nDocument {i} TF-IDF:")
    for term, score in sorted(doc_tfidf.items(), key=lambda x: -x[1]):
        print(f"{term:<12} : {score:.4f}")