#!/usr/bin/env python.

"""
CS4248 ASSIGNMENT 2 Template


"""

# Import libraries.  Add any additional ones here.
# Generally, system libraries precede others.
import pandas as pd
import random
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from nltk.stem import WordNetLemmatizer, PorterStemmer
import string
from nltk.corpus import stopwords
import nltk
from sklearn.model_selection import train_test_split
from statistics import mean

# Download NLTK data
nltk.download('wordnet')
nltk.download('stopwords')

_NAME = "XiaoYan"
_STUDENT_NUM = 'E0902032'

def preprocess_text(text):
    ''' Custom text preprocessing function '''
    stop_words = set(stopwords.words('english'))
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()  # Case-folding to all lowercase
    words = text.split()
    words = [word for word in words if word not in stop_words and word.isalpha()]
    return ' '.join(words)

def train_model(model, X_train, y_train):
    ''' train your model based on the training data '''
    model.fit(X_train, y_train)

def predict(model, X_test):
    ''' make your prediction here '''
    return model.predict(X_test)

def generate_result(test, y_pred, filename):
    ''' generate csv file base on the y_pred '''
    test['Verdict'] = pd.Series(y_pred)
    test.drop(columns=['Text'], inplace=True)
    test.to_csv(filename, index=False)

def main():
    ''' load train, val, and test data '''
    # Load the dataset
    dataset = pd.read_csv('Sentiment_dataset.csv')
    
    # Drop 'text' and 'url' columns
    dataset.drop(columns=['text', 'url'], inplace=True)
    
    # Convert 'sentiment' column to integer
    dataset['sentiment'] = dataset['sentiment'].astype(int)

    print("\nFirst 5 Rows of the Dataset:")
    print(dataset.head())
    
    # Split the data into train and test sets
    train, test = train_test_split(dataset, test_size=0.2, random_state=42)  # 20% test size

    # Number of iterations
    iterations = 5

    train_scores = []
    test_scores = []

    for i in range(iterations):
        # Split the data into train and test sets
        train, test = train_test_split(dataset, test_size=0.2, random_state=random.randint(0, 10000))

        # Assign train and test data
        X_train = train['news_title'] + " " + train['reddit_title']
        y_train = train['sentiment']
        X_test = test['news_title'] + " " + test['reddit_title']
        y_test = test['sentiment']

        # Model: Neural Network with TfidfVectorizer
        model_nn_tfidf = make_pipeline(
            TfidfVectorizer(stop_words='english', lowercase=True),
            MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
        )

        model = model_nn_tfidf

        # Train the model
        train_model(model, X_train, y_train)

        # Evaluate on training data
        y_train_pred = predict(model, X_train)
        train_score = f1_score(y_train, y_train_pred, average='macro')
        train_scores.append(train_score)

        # Evaluate on test data
        y_test_pred = predict(model, X_test)
        test_score = f1_score(y_test, y_test_pred, average='macro')
        test_scores.append(test_score)

        print(f"Iteration {i + 1}: Train Score = {train_score}, Test Score = {test_score}")

    # Calculate average scores
    avg_train_score = mean(train_scores)
    avg_test_score = mean(test_scores)

    print(f"\nAverage Train Score: {avg_train_score}")
    print(f"Average Test Score: {avg_test_score}")

    
# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()
