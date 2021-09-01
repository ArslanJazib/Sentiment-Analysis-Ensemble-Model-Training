import time
import joblib
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier

class SupportVectorMachine:
    """Using support vector machine for traning a model"""

    def __init__(self, features, tweets):
        self.tweets = tweets
        self.features=features
        # Splitting each sample data set into training and testing sets with the ratio of 80:20
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.features,self.tweets['target'],test_size=0.2,stratify=tweets['target'])
        self.x_train = self.x_train[:,1:]
        self.x_test = self.x_test[:,1:]
   
    def classification_report(self,predictions):
        
        # Checking the performance of the algorithms
        for prediction in predictions:
            report = classification_report(self.y_test, prediction, target_names=class_names)
            print('positive: ', report['1'])
            print('negative: ', report['0'])


    def svm_classifiy(self):
        # Perform classification with SVM, kernel=linear
        svm_classifier_linear = svm.SVC(kernel='linear')
        svm_classifier_linear.fit(self.x_train,self.y_train)
        joblib.dump(svm_classifier_linear, 'Svm_Classifier_Linear.pkl')


    def svm_ensembleClassifier(self):
        
        features=self.feature_generator()

        # Create a dictionary of our pre trained SVM models
        svm1 = joblib.load('SVM_Linear_Classifier1.pkl')
        svm2 = joblib.load('SVM_Linear_Classifier2.pkl')
        svm3 = joblib.load('SVM_Linear_Classifier3.pkl')
        svm4 = joblib.load('SVM_Linear_Classifier4.pkl')
        svm5 = joblib.load('SVM_Linear_Classifier5.pkl')
        svm6 = joblib.load('SVM_Linear_Classifier6.pkl')
        svm7 = joblib.load('SVM_Linear_Classifier7.pkl')
        svm8 = joblib.load('SVM_Linear_Classifier8.pkl')

        # Getting predictions from each base classifier
        pred1=svm1.predict(self.x_test)
        pred2=svm2.predict(self.x_test)
        pred3=svm3.predict(self.x_test)
        pred4=svm4.predict(self.x_test)
        pred5=svm5.predict(self.x_test)
        pred6=svm6.predict(self.x_test)
        pred7=svm7.predict(self.x_test)
        pred8=svm8.predict(self.x_test)

        # Votting Ensemble
        final_pred = np.array([])
        for i in range(0,len(self.x_test)):
            final_pred = np.append(final_pred, mode([pred1[i], pred2[i], pred3[i], pred4[i], pred5[i], pred6[i], pred7[i], pred8[i]]))

        # A list of all predictions to be used for generating classification report for base classifier and majority votted ouput
        predictions=[pred1,pred2,pred3,pred4,pred4,pred5,pred6,pred7,pred8,final_pred]

        # Printing Classification Reports
        self.classification_report(predictions)