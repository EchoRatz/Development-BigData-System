import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# One-time: download the VADER lexicon
nltk.download("vader_lexicon")

# 1. Load your file from repo's out folder
df = pd.read_csv("redditCrawler\out\posts_preprocessed.csv")

# 2. Initialize VADER
sia = SentimentIntensityAnalyzer()

# 3. Define classification function
def get_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return "NEU"
    score = sia.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "POS"
    elif score <= -0.05:
        return "NEG"
    else:
        return "NEU"

# 4. Combine title + selftext for richer sentiment analysis
df["text_for_sentiment"] = df[["title", "text_clean"]].fillna("").agg(" ".join, axis=1)

# 5. Apply sentiment classifier
df["sentiment"] = df["text_for_sentiment"].apply(get_sentiment)

# 6. Save result back into the out folder
df.to_csv("redditCrawler/out/001_with_sentiment.csv", index=False)

print("✅ Finished! File saved as out/001_with_sentiment.csv")
