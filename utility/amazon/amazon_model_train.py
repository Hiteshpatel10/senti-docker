import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import re
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score 

def clean_text(text):
    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # Remove extra whitespaces
    text = re.sub(r"\s+", " ", text)
    
    # Convert text to lowercase
    text = text.lower()
    
    return text

def amazonModelTrain():

    df = pd.read_csv('../../data/am1-train.csv')
    
    print(len(df))

    sentiments = SentimentIntensityAnalyzer()
    df["positive"] = df['content'].apply(lambda review: sentiments.polarity_scores(str(review))["pos"])
    df["negative"] = df['content'].apply(lambda review: sentiments.polarity_scores(str(review))["neg"])
    df["neutral"] = df['content'].apply(lambda review: sentiments.polarity_scores(str(review))["neu"])
    df["compound"] = df['content'].apply(lambda review: sentiments.polarity_scores(str(review))["compound"])

    score = df["compound"].values
    sentiment = []
    for i in score:
        if i >= 0.05 :
            sentiment.append(1)
        elif i <= -0.05 :
            sentiment.append(-1)
        else:
            sentiment.append(0)
    df["sentiment"] = sentiment

    vectorizer = TfidfVectorizer(max_features=2500)
    x = vectorizer.fit_transform(df['content'].apply(lambda x: np.str_(x)))
    y = df['sentiment']

     # Split into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


    # Naive Bayes Model
    nb_model = MultinomialNB()
    nb_scores = cross_validate(nb_model, x, y, cv=5, return_train_score=True)
    nb_model.fit(x_train, y_train)
    nb_pred = nb_model.predict(x_test)
    print("Naive Bayes Model Precision: ", precision_score(y_test, nb_pred, average='weighted'))
    print("Naive Bayes Model Recall: ", recall_score(y_test, nb_pred, average='weighted'))
    print("Naive Bayes Model F1 Score: ", f1_score(y_test, nb_pred, average='weighted'))
    print("Naive Bayes Model Accuracy: ", nb_scores['test_score'].mean())
    

    # Train Logistic Regression Model
    lr_model = LogisticRegression(max_iter=1000)
    lr_scores = cross_validate(lr_model, x, y, cv=5, return_train_score=True)
    lr_model.fit(x_train, y_train)
    lr_pred = lr_model.predict(x_test)
    print("Logistic Regression Model Precision: ", precision_score(y_test, lr_pred, average='weighted'))
    print("Logistic Regression Model Recall: ", recall_score(y_test, lr_pred, average='weighted'))
    print("Logistic Regression Model F1 Score: ", f1_score(y_test, lr_pred, average='weighted'))
    print("Logistic Regression Model Accuracy: ", lr_scores['test_score'].mean())

    # Train KNN Model
    knn_model = KNeighborsClassifier(n_neighbors=4)
    knn_scores = cross_validate(knn_model, x, y, cv=5, return_train_score=True)
    knn_model.fit(x_train, y_train)
    knn_pred = knn_model.predict(x_test)
    print("KNN Model Precision: ", precision_score(y_test, knn_pred, average='weighted'))
    print("KNN Model Recall: ", recall_score(y_test, knn_pred, average='weighted'))
    print("KNN Model F1 Score: ", f1_score(y_test, knn_pred, average='weighted'))
    print("KNN Model Accuracy: ", knn_scores['test_score'].mean())

    # SVM Model
    svm_model = SVC(kernel='linear')
    svm_scores = cross_validate(svm_model, x, y, cv=5, return_train_score=True)
    svm_model.fit(x_train, y_train)
    svm_pred = svm_model.predict(x_test)
    print("SVM Model Precision: ", precision_score(y_test, svm_pred, average='weighted'))
    print("SVM Model Recall: ", recall_score(y_test, svm_pred, average='weighted'))
    print("SVM Model F1 Score: ", f1_score(y_test, svm_pred, average='weighted'))
    print("SVM Model Accuracy: ", svm_scores['test_score'].mean())

    
if __name__ == "__main__":
    amazonModelTrain()




    # Naive Bayes Model
    nb_model = MultinomialNB()
    nb_scores = cross_validate(nb_model, x, y, cv=5, return_train_score=True)
    nb_model.fit(x_train, y_train)
    nb_pred = nb_model.predict(x_test)

    # Train Logistic Regression Model
    lr_model = LogisticRegression(max_iter=1000)
    lr_scores = cross_validate(lr_model, x, y, cv=5, return_train_score=True)
    lr_model.fit(x_train, y_train)
    lr_pred = lr_model.predict(x_test)

    # Train KNN Model
    knn_model = KNeighborsClassifier(n_neighbors=4)
    knn_scores = cross_validate(knn_model, x, y, cv=5, return_train_score=True)
    knn_model.fit(x_train, y_train)
    knn_pred = knn_model.predict(x_test)

    # SVM Model
    svm_model = SVC(kernel='linear')
    svm_scores = cross_validate(svm_model, x, y, cv=5, return_train_score=True)
    svm_model.fit(x_train, y_train)
    svm_pred = svm_model.predict(x_test)

