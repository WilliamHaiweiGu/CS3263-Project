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
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

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
    text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    text = re.sub(r"[^a-zA-z.!?'0-9]", ' ', text)
    text = re.sub('\t', ' ', text)
    text = re.sub(r" +", ' ', text)
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stopwords and non-alphabetic tokens
    tokens = [tok for tok in tokens if tok.isalpha() and tok not in stop_words]
    # Lemmatize each token
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
    return tokens

def main():
    df = pd.read_csv('/Users/jerry/Documents/GitHub/CS3263-Project/Sentiment_dataset.csv')
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
    f1_scores_gnb = []
    acc_scores = []
    acc_scores_gnb = []

    for i in range(iterations):
        # Split the data into training and testing sets, ONLY USE THE TEXT COLUMN
        X_train, X_test, y_train, y_test = train_test_split(X_text, labels, test_size=0.2, random_state=42+i, stratify=labels)

        # Handle class imbalance with class weights
        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        sample_weights = np.array([class_weight_dict[label] for label in y_train])

        #GaussianNB expects dense arrays, so convert from sparse matrix.
        X_train_dense = X_train.toarray()
        X_test_dense = X_test.toarray()

        # Train MultinomialNB model
        model = MultinomialNB()
        model.fit(X_train, y_train, sample_weight=sample_weights)

        # Train GaussianNB model
        gnb_model = GaussianNB()
        gnb_model.fit(X_train_dense, y_train, sample_weight=sample_weights)

        # Evaluate the model
        y_pred = model.predict(X_test)
        y_pred_gnb = gnb_model.predict(X_test_dense)
        acc = accuracy_score(y_test, y_pred)
        acc_gnb = accuracy_score(y_test, y_pred_gnb)
        # Calculate F1 score
        f1 = f1_score(y_test, y_pred, average='macro')
        f1_gnb = f1_score(y_test, y_pred_gnb, average='macro')
        cf_matrix = confusion_matrix(y_test, y_pred)
        cf_matrix_gnb = confusion_matrix(y_test, y_pred_gnb)
        print("MultinomialNB Confusion Matrix:\n")
        print(cf_matrix)
        print("GaussianNB Confusion Matrix:\n")
        print(cf_matrix_gnb)
        acc_scores.append(acc)
        acc_scores_gnb.append(acc_gnb)
        f1_scores.append(f1)
        f1_scores_gnb.append(f1_gnb)

        print(f"Iteration {i + 1}: MultinomialNB Accuracy = {acc:.4f}, MultinomialNB Macro F1 Score = {f1:.4f}")
        print(f"Iteration {i + 1}: GaussianNB Accuracy = {acc_gnb:.4f}, GaussianNB Macro F1 Score = {f1_gnb:.4f}")
        #print(f"Iteration {i + 1}: Test F1 Score = {f1_scores}, Test Accuracy = {acc_scores}")
    
    # Calculate average scores
    avg_f1 = np.mean(f1_scores)
    avg_f1_gnb = np.mean(f1_scores_gnb)
    avg_acc = np.mean(acc_scores)
    avg_acc_gnb = np.mean(acc_scores_gnb)
    print(f"Average Macro F1 Score (MultinomialNB): {avg_f1:.4f}")
    print(f"Average Accuracy (MultinomialNB): {avg_acc:.4f}")
    print(f"Average Macro F1 Score (GaussianNB): {avg_f1_gnb:.4f}")
    print(f"Average Accuracy (GaussianNB): {avg_acc_gnb:.4f}")


if __name__ == "__main__":
    main()
