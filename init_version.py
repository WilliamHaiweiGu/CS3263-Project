#!/usr/bin/env python.

"""
CS4248 ASSIGNMENT 2 Template


"""

# Import libraries.  Add any additional ones here.
# Generally, system libraries precede others.
import pandas as pd
import random
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
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
from sklearn.metrics import confusion_matrix
import seaborn as sns   
import matplotlib.pyplot as plt
import re
import numpy as np

# Download NLTK data
nltk.download('wordnet')
nltk.download('stopwords')

_NAME = "XiaoYan"
_STUDENT_NUM = 'E0902032'

def clean_text(text: str) -> str:
    text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    text = re.sub(r"[^a-zA-z.!?'0-9]", ' ', text)
    text = re.sub('\t', ' ', text)
    return re.sub(r" +", ' ', text)

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
    
    # Use 'text' as the predictor and drop unnecessary columns
    dataset.drop(columns=['url', 'news_title', 'reddit_title'], inplace=True)

    # apply text preprocessing
    dataset['text'] = dataset['text'].apply(clean_text)
    dataset['text'] = dataset['text'].apply(preprocess_text)
    
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
        X_train = train['text']
        y_train = train['sentiment']
        X_test = test['text']
        y_test = test['sentiment']

        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        sample_weights = np.array([class_weight_dict[label] for label in y_train])

        # Model: Neural Network with TfidfVectorizer
        model_nb_cv = make_pipeline(
            CountVectorizer(stop_words='english', lowercase=True),
            MultinomialNB(alpha=1)  
        )

        # Model 2: Naive Bayes with TfidfVectorizer
        model_nb_tfidf = make_pipeline(
            TfidfVectorizer(stop_words='english', lowercase=True),
            MultinomialNB(alpha=1)  
        )
        # train score: 0.525
        # test score: 0.544

        # Model 3: Logistic Regression with CountVectorizer
        model_lr_cv = make_pipeline(
            CountVectorizer(stop_words='english', lowercase=True),
            LogisticRegression(max_iter=1000, C=0.5)  
        )
        # train score: 0.758
        # test score: 0.705

        # Model 4: Logistic Regression with TfidfVectorizer
        model_lr_tfidf = make_pipeline(
            TfidfVectorizer(stop_words='english', lowercase=True),
            LogisticRegression(max_iter=1000, C=0.5)  
        )
        # train score: 0.631
        # test score: 0.650


        # Model 5: Neural Network with CountVectorizer
        model_nn_cv1 = make_pipeline(
            CountVectorizer(stop_words='english', lowercase=True),
            MLPClassifier(hidden_layer_sizes=(150,), max_iter=500, alpha=0.001)  
        )
        # train score: 0.994
        # test score: 0.719

        # Model 6: Neural Network with CountVectorizer with different hyperparameters
        model_nn_cv_2 = make_pipeline(
            CountVectorizer(stop_words='english', lowercase=True),
            MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, alpha=0.001)
        )
        # train score: 0.994
        # test score: 0.741

        # Model 7: Neural Network with CountVectorizer with more hyperparameters
        model_nn_cv_3 = make_pipeline(
            CountVectorizer(stop_words='english', lowercase=True),
            MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, alpha=0.01, learning_rate_init=0.001, early_stopping=True)
        )
        # train score: 0.779
        model = model_nb_cv

        # Train the model
        train_model(model, X_train, y_train)

        # Evaluate on training data
        y_train_pred = predict(model, X_train)
        train_score = f1_score(y_train, y_train_pred, average='macro')
        train_accuracy = (y_train == y_train_pred).mean()
        train_scores.append((train_score, train_accuracy))

        # Evaluate on test data
        y_test_pred = predict(model, X_test)
        test_score = f1_score(y_test, y_test_pred, average='macro')
        test_accuracy = (y_test == y_test_pred).mean()
        test_scores.append((test_score, test_accuracy))

        print(f"Iteration {i + 1}: Train F1 Score = {train_score}, Train Accuracy = {train_accuracy}, "
              f"Test F1 Score = {test_score}, Test Accuracy = {test_accuracy}")
        
        # print confusion matrix
        
        cm = confusion_matrix(y_test, y_test_pred)

        print(cm)

        # Calculate average scores
        avg_train_score = mean([score[0] for score in train_scores])
        avg_train_accuracy = mean([score[1] for score in train_scores])
        avg_test_score = mean([score[0] for score in test_scores])
        avg_test_accuracy = mean([score[1] for score in test_scores])

    print(f"\nAverage Train F1 Score: {avg_train_score}, Average Train Accuracy: {avg_train_accuracy}")
    print(f"Average Test F1 Score: {avg_test_score}, Average Test Accuracy: {avg_test_accuracy}")

    
# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()
