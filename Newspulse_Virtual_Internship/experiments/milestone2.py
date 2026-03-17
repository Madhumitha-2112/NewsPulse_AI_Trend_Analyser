import pandas as pd
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')

file_path = r"C:\Users\Welcome\Desktop\Newspulse_Virtual_Internship\multi_genre_news_dataset.csv"
df = pd.read_csv(file_path)

df.columns = df.columns.str.strip().str.lower()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)      
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    return text

df["cleaned_text"] = df["title"].apply(clean_text)

df.to_csv("cleaned_news_m2.csv", index=False)

print("Text cleaning completed!")


import nltk
from nltk.corpus import stopwords

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    
    tokens = text.split()
    
    filtered_tokens = [
        word for word in tokens
        if word.isalpha() and
        word not in stop_words and
        len(word) > 2
    ]
    
    return " ".join(filtered_tokens)
    
    return " ".join(filtered_tokens)

df["processed_text"] = df["cleaned_text"].apply(preprocess_text)

df.to_csv("processed_news.csv", index=False)

print("Tokenization and stopword removal completed!")


from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

vectorizer = TfidfVectorizer(max_features=1000)

tfidf_matrix = vectorizer.fit_transform(df["processed_text"])

print("TF-IDF Matrix Shape:", tfidf_matrix.shape)


feature_names = vectorizer.get_feature_names_out()
mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
top_indices = mean_scores.argsort()[-10:][::-1]

print("\nTop 10 Trending Keywords:\n")
for index in top_indices:
    print(feature_names[index])


from sklearn.decomposition import LatentDirichletAllocation


lda_model = LatentDirichletAllocation(
    n_components=3,
    random_state=42
)

lda_model.fit(tfidf_matrix)

print("\nGenerated Topics:\n")

feature_names = vectorizer.get_feature_names_out()

for topic_idx, topic in enumerate(lda_model.components_):
    print(f"\nTopic {topic_idx + 1}:")
    
    top_indices = topic.argsort()[-10:][::-1]
    
    for index in top_indices:
        print(feature_names[index], end=", ")
    
    print()


from textblob import TextBlob

def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["processed_text"].apply(get_sentiment)

print("\nSentiment Analysis Completed!\n")

print("Sentiment Distribution:\n")
print(df["sentiment"].value_counts())

df.to_csv("news_with_sentiment.csv", index=False)