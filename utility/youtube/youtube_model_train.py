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
from joblib import dump

# Load YouTube comments dataset
df = pd.read_csv('../../data/youtube.csv')

# Compute sentiment scores using VADER
sentiments = SentimentIntensityAnalyzer()
df["positive"] = df['comment'].apply(lambda review: sentiments.polarity_scores(str(review))["pos"])
df["negative"] = df['comment'].apply(lambda review: sentiments.polarity_scores(str(review))["neg"])
df["neutral"] = df['comment'].apply(lambda review: sentiments.polarity_scores(str(review))["neu"])
df["compound"] = df['comment'].apply(lambda review: sentiments.polarity_scores(str(review))["compound"])

# Classify comments as positive, negative, or neutral based on compound sentiment score
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
x = vectorizer.fit_transform(df['comment'].apply(lambda x: np.str_(x)))
y = df['sentiment']
dump(vectorizer, '../../model/youtube/vectorizer.joblib')

    # Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)



# Naive Bayes Model
nb_model = MultinomialNB()
nb_scores = cross_validate(nb_model, x, y, cv=5, return_train_score=True)
nb_model.fit(x_train, y_train)
nb_pred = nb_model.predict(x_test)
dump(nb_model, '../../model/youtube/naive_bayes_model.joblib')
print("Naive Bayes Model Precision: ", precision_score(y_test, nb_pred, average='weighted'))
print("Naive Bayes Model Recall: ", recall_score(y_test, nb_pred, average='weighted'))
print("Naive Bayes Model F1 Score: ", f1_score(y_test, nb_pred, average='weighted'))
print("Naive Bayes Model Accuracy: ", nb_scores['test_score'].mean())



# Train Logistic Regression Model
lr_model = LogisticRegression(max_iter=1000)
lr_scores = cross_validate(lr_model, x, y, cv=5, return_train_score=True)
lr_model.fit(x_train, y_train)
lr_pred = lr_model.predict(x_test)
dump(nb_model, '../../model/youtube/lr_model.joblib')
print("Logistic Regression Model Precision: ", precision_score(y_test, lr_pred, average='weighted'))
print("Logistic Regression Model Recall: ", recall_score(y_test, lr_pred, average='weighted'))
print("Logistic Regression Model F1 Score: ", f1_score(y_test, lr_pred, average='weighted'))
print("Logistic Regression Model Accuracy: ", lr_scores['test_score'].mean())

# Train KNN Model
knn_model = KNeighborsClassifier(n_neighbors=4)
knn_scores = cross_validate(knn_model, x, y, cv=5, return_train_score=True)
knn_model.fit(x_train, y_train)
knn_pred = knn_model.predict(x_test)
dump(nb_model, '../../model/youtube/knn_model.joblib')
print("KNN Model Precision: ", precision_score(y_test, knn_pred, average='weighted'))
print("KNN Model Recall: ", recall_score(y_test, knn_pred, average='weighted'))
print("KNN Model F1 Score: ", f1_score(y_test, knn_pred, average='weighted'))
print("KNN Model Accuracy: ", knn_scores['test_score'].mean())

# SVM Model
svm_model = SVC(kernel='linear')
svm_scores = cross_validate(svm_model, x, y, cv=5, return_train_score=True)
svm_model.fit(x_train, y_train)
svm_pred = svm_model.predict(x_test)
dump(nb_model, '../../model/youtube/svm_model.joblib')
print("SVM Model Precision: ", precision_score(y_test, svm_pred, average='weighted'))
print("SVM Model Recall: ", recall_score(y_test, svm_pred, average='weighted'))
print("SVM Model F1 Score: ", f1_score(y_test, svm_pred, average='weighted'))
print("SVM Model Accuracy: ", svm_scores['test_score'].mean())
