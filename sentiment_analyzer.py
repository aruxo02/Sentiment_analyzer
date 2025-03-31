import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

# 1. Load data from texts.csv
df = pd.read_csv("texts.csv")

# 2. Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# 3. Sentiment classification function
def classify_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# 4. Apply analysis
df["sentiment"] = df["text"].apply(classify_sentiment)

# 5. Save results
df.to_csv("results.csv", index=False)

# 6. Print results
print(df)

# 7. Plot chart
counts = df["sentiment"].value_counts()
plt.figure(figsize=(6,6))
plt.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=140)
plt.title("Sentiment Analysis Results")
plt.axis("equal")
plt.show()
