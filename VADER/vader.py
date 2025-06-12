import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
text = "I love Natural Language Processing!"

scores = analyzer.polarity_scores(text)

print(scores)