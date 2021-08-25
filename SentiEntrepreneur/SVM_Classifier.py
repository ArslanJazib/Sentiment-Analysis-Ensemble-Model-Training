import time
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
import joblib


class SupportVectorMachine:
    """Using support vector machine for traning a model"""
    
    def __init__(self, tweets):
        self.tweets = tweets
        #splitting each sample data set into training and testing sets with the ratio of 80:20
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(tweets['lemmatized_text'],tweets['target'],test_size=0.2,stratify=tweets['target'])


    def feature_generator(self):
        # create the transform for term frequency document
        vectorizer = TfidfVectorizer(min_df = 5,max_df = 0.8,sublinear_tf = True,use_idf = True)
        # Transforming training and testing data into vectors
        train_vectors = vectorizer.fit_transform(self.x_train)
        # Term Frequency Document Matrix can be used to visualzie vectorization
        train_matrix = pd.DataFrame(train_vectors.toarray().transpose(), index = vectorizer.get_feature_names())
        test_vectors = vectorizer.transform(self.x_test)
        # Term Frequency Document Matrix can be used to visualzie vectorization
        test_matrix = pd.DataFrame(test_vectors.toarray().transpose(), index = vectorizer.get_feature_names())
        # To display the number of times a word is occured in the dataset
        return [train_vectors,test_vectors]

    def svm_classifiy(self):
        vectors=self.feature_generator()
        # Perform classification with SVM, kernel=linear
        svm_classifier_linear = svm.SVC(kernel='linear')
        t0 = time.time()
        # To check the class imbalance in the training sample
        print("Class distribution between positive & negative")
        print(self.y_train.value_counts(normalize=True))
        svm_classifier_linear.fit(vectors[0], self.y_train)
        # Save the model as a pickle in a file
        joblib.dump(svm_classifier_linear, 'SVM_Linear_Classifier8.pkl')

        #t1 = time.time()
        #prediction_linear = svm_classifier_linear.predict(vectors[1])
        #t2 = time.time()
        #time_linear_train = t1-t0
        #time_linear_predict = t2-t1
        # results
        #print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
        # Checking the performance of the algorithm
        #report = classification_report(self.y_test, prediction_linear, output_dict=True)
        #print('positive: ', report['1'])
        #print('negative: ', report['0'])


