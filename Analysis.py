import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load preprocessed data
play_news = pd.read_csv('PlayNews_Preprocessed.csv')
noname_news = pd.read_csv('NoNameNews_Preprocessed.csv')
ip_data = pd.read_csv('NoNameIP_Preprocessed.csv')

# TF-IDF Analysis
def perform_tfidf(news_df):
    """Perform TF-IDF analysis and extract top keywords."""
    vectorizer = TfidfVectorizer(max_features=10)
    tfidf_matrix = vectorizer.fit_transform(news_df['Cleaned_Content'])
    keywords = vectorizer.get_feature_names_out()
    return keywords

play_keywords = perform_tfidf(play_news)
noname_keywords = perform_tfidf(noname_news)

# Sentiment Analysis
def perform_sentiment_analysis(news_df):
    """Analyze sentiment of headlines."""
    sentiments = []
    for headline in news_df['Headline']:
        polarity = TextBlob(headline).sentiment.polarity
        sentiments.append(polarity)
    news_df['Sentiment'] = sentiments
    return news_df

play_news = perform_sentiment_analysis(play_news)
noname_news = perform_sentiment_analysis(noname_news)

# Save analyzed datasets
play_news.to_csv('PlayNews_Analyzed.csv', index=False)
noname_news.to_csv('NoNameNews_Analyzed.csv', index=False)

# Geolocation Analysis
def process_geolocation(ip_df):
    """Aggregate and rank geolocation data by frequency."""
    geo_counts = ip_df['geolocation'].value_counts().head(20)
    return geo_counts

top_geo_counts = process_geolocation(ip_data)

# Save geolocation results
top_geo_counts.to_csv('TopGeolocations.csv')

print("Analysis complete. Results saved.")
