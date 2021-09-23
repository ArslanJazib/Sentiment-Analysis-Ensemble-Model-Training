import joblib
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class NaiveBayes_Classifier(object):
    """To create an Naive Bayes Classifier and generate classification report"""

    def __init__(self, features, tweets):
        self.tweets = tweets
        self.features=features
        # Splitting each sample data set into training and testing sets with the ratio of 80:20
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.features,self.tweets['target'],test_size=0.2,stratify=tweets['target'])
        self.x_train = self.x_train[:,1:]
        self.x_test = self.x_test[:,1:]
   
    def classification_report(self):
        
        # Pre-Trained Naive Bayes Classifier
        svm_classifier = joblib.load('Resources/NaiveBayes_Classifier.pkl')

        # Vectorizer used for training
        vectorizer = joblib.load('Resources/SVM_Vectorizer.pkl')
        
        # Tranforming test data using count vectorizer
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.tweets['lemmatized_text'],self.tweets['target'],test_size=0.2,stratify=self.tweets['target'])
        self.x_test=vectorizer.transform(self.x_test)

        # Preditcing test data values
        prediction=svm_classifier.predict(self.x_test)

        # Classification report
        report = classification_report(self.y_test, prediction)

        print(report)

    def naive_Classifier(self):
        naive_model = MultinomialNB()
        naive_model.fit(self.x_train,self.y_train)
        joblib.dump(naive_model, 'Resources/NaiveBayes_Classifier.pkl')




