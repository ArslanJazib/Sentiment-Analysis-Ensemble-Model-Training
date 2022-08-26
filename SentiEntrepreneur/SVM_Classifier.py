import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



class SupportVectorMachine:
    """To create an ensemble SVM Classifier and generate classification report"""

    def __init__(self, features, tweets):
        self.tweets = tweets
        self.features=features
        # Splitting each sample data set into training and testing sets with the ratio of 80:20
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.features,self.tweets['target'],test_size=0.2,stratify=tweets['target'])
        self.x_train = self.x_train[:,1:]
        self.x_test = self.x_test[:,1:]
   
    def classification_report(self):
        
        # Pre-Trained SVM Classifier
        svm_classifier = joblib.load('Resources/Svm_Classifier_linear.pkl')

        # Vectorizer used for training
        vectorizer = joblib.load('Resources/SVM_Vectorizer.pkl')
        
        # Tranforming test data using count vectorizer
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.tweets['lemmatized_text'],self.tweets['target'],test_size=0.2,stratify=self.tweets['target'])
        self.x_test=vectorizer.transform(self.x_test)

        # Preditcing test data values
        prediction=svm_classifier.predict(self.x_test)
        joblib.dump(prediction, 'Resources/SVM_Predictions.pkl')

        # Classification report
        report = classification_report(self.y_test, prediction)

        print(report)
        
        print(confusion_matrix(self.y_test, prediction))


    def svm_ensembleClassifier(self):
        n_estimators = 10
        svm_classifier_linear = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear',verbose=True), max_samples=1.0 / n_estimators, n_estimators=n_estimators,bootstrap=False))
        svm_classifier_linear.fit(self.x_train,self.y_train)
        joblib.dump(svm_classifier_linear, 'Resources/Svm_Classifier_linear.pkl')



        
