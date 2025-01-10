import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load datasets
ip_data = pd.read_csv('NoNameIP.csv')
play_news = pd.read_excel('PlayNews.xlsx')
noname_news = pd.read_excel('NoNameNews.xlsx')

# Define preprocessing functions
def clean_text(text):
    """Normalize and clean text data."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters and punctuation
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(tokens)

# Preprocess Play News
def preprocess_news(news_df):
    news_df['Cleaned_Content'] = news_df['Content'].apply(clean_text)
    return news_df

play_news = preprocess_news(play_news)
noname_news = preprocess_news(noname_news)

# Save preprocessed datasets
play_news.to_csv('PlayNews_Preprocessed.csv', index=False)
noname_news.to_csv('NoNameNews_Preprocessed.csv', index=False)
ip_data.to_csv('NoNameIP_Preprocessed.csv', index=False)

print("Preprocessing complete. Cleaned datasets saved.")
