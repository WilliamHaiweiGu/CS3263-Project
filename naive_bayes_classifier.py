# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def main():
    dataset = pd.read_csv('Sentiment_dataset.csv')
    dataset.drop(['url'], axis=1)

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    dataset['clean_text'] = dataset['text'].apply(lambda x: ' '.join([word for word in word_tokenize(x.lower()) if word.isalpha() and word not in stop_words]))

    labels = dataset['sentiment'].astype(int)
    X = dataset['news_title'] + " " + dataset['reddit_title'] + "" + dataset['clean_text']

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)


if __name__ == "__main__":
    main()
