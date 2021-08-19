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

class SupportVectorMachine:
    """Using support vector machine for traning a model"""
    
    def __init__(self, tweets):
        self.tweets = tweets
        #splitting the data set into training and testing sets with the ratio of 80:20
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(tweets['lemmatized_text'],tweets['target'],test_size=0.2,stratify=tweets['target'])

    def feature_generator(self):
        # create the transform for term frequency document
        vectorizer = TfidfVectorizer(min_df = 5,max_df = 0.8,sublinear_tf = True,use_idf = True)
        # Transforming training and testing data into vectors
        train_vectors = vectorizer.fit_transform(self.x_train)
        train_matrix = pd.DataFrame(train_vectors.toarray().transpose(), index = vectorizer.get_feature_names())
        test_vectors = vectorizer.fit_transform(self.x_test)
        test_matrix = pd.DataFrame(test_vectors.toarray().transpose(), index = vectorizer.get_feature_names())
        # To display the number of times a word is occured in the dataset
        # print('vocabulary: ', vectorizer.vocabulary_)
        return train_vectors

    def svm_classifiy(self):
        train_vectors=self.feature_generator()
        # Perform classification with SVM, kernel=linear
        classifier_linear = svm.SVC(kernel='linear')
        t0 = time.time()
        print(self.y_train.value_counts(normalize=True))
        classifier_linear.fit(train_vectors, self.y_train)
        t1 = time.time()
        prediction_linear = classifier_linear.predict(test_vectors)
        t2 = time.time()
        time_linear_train = t1-t0
        time_linear_predict = t2-t1
        # results
        print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
        # Checking the performance of the algorithm
        report = classification_report(self.y_test, classifier_linear, output_dict=True)
        print('positive: ', report[4])
        print('negative: ', report[0])


