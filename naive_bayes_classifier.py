# Import libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """Clean and tokenize text string into lemmatized tokens."""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    # Lowercase the text
    text = text.lower()
    # Remove punctuation and digits (keep only letters and whitespace)
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stopwords and non-alphabetic tokens
    tokens = [tok for tok in tokens if tok.isalpha() and tok not in stop_words]
    # Lemmatize each token
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
    return tokens

def main():
    df = pd.read_csv('Sentiment_dataset.csv')
    df.drop(['url'], axis=1)

    #df['clean_text'] = df['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(tok) for tok in [word for word in word_tokenize(re.sub(r'[^a-z\s]', ' ', x.lower())) if word.isalpha() and word not in stop_words]]))

    tfidf_news = TfidfVectorizer(tokenizer=preprocess_text, preprocessor=None, lowercase=False)
    tfidf_reddit = TfidfVectorizer(tokenizer=preprocess_text, preprocessor=None, lowercase=False)
    tfidf_text = TfidfVectorizer(tokenizer=preprocess_text, preprocessor=None, lowercase=False)

    labels = df['sentiment'].astype(int)
    X_news = tfidf_news.fit_transform(df['news_title'])
    X_reddit = tfidf_reddit.fit_transform(df['reddit_title'])
    X_text = tfidf_text.fit_transform(df['text'])
    X_all = hstack([X_news, X_reddit, X_text])

    iterations = 10
    f1_scores = []
    acc_scores = []

    for i in range(iterations):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_all, labels, test_size=0.2, random_state=i, stratify=labels)

        # Handle class imbalance with class weights
        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        sample_weights = np.array([class_weight_dict[label] for label in y_train])

        # Train MultinomialNB model
        model = MultinomialNB()
        model.fit(X_train, y_train, sample_weight=sample_weights)

        # Evaluate the model
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        acc_scores.append(acc)
        f1_scores.append(f1)

        print(f"Iteration {i + 1}: Accuracy = {acc:.4f}, F1 Score = {f1:.4f}")
        print(f"Iteration {i + 1}: Test F1 Score = {f1_scores}, Test Accuracy = {acc_scores}")
    
    # Calculate average scores
    avg_f1 = np.mean(f1_scores)
    avg_acc = np.mean(acc_scores)
    print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"Average Accuracy: {avg_acc:.4f}")


if __name__ == "__main__":
    main()
