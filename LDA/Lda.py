# Example: Using LDiA to find topics in IT incident reports

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 🗂️ Example IT incident reports
documents = [
    "Database connection timed out during peak hours.",
    "Website is showing a 502 Bad Gateway error.",
    "User cannot authenticate via single sign-on.",
    "SSL certificate expired on production server."
]

# ✅ Convert text to Bag-of-Words
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

#  Run Latent Dirichlet Allocation (LDiA)
lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(X)

#  Display top words in each topic
terms = vectorizer.get_feature_names_out()

for idx, topic in enumerate(lda.components_):
    print(f"\n Topic {idx + 1}:")
    top_terms_idx = topic.argsort()[-5:][::-1]
    top_terms = [terms[i] for i in top_terms_idx]
    print("Top keywords:", ", ".join(top_terms))

#  Show topic distribution for each document
doc_topics = lda.transform(X)

print("\n Document-Topic Distribution:")
for idx, doc in enumerate(documents):
    topic_mix = ", ".join([f"Topic {i+1}: {weight:.2f}" for i, weight in enumerate(doc_topics[idx])])
    print(f"Doc {idx+1}: {topic_mix}")
