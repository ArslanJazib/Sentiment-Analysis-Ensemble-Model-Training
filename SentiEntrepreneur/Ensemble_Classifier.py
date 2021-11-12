import os
import sys
import joblib
import itertools
from statistics import mode
import numpy as np
from keras.models import load_model
from sklearn.ensemble import VotingClassifier
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer


class Ensemble_Classifier(object):
    """ This class will use majority votting to form an ensemble classifier using SVM, Neural Network, Naive Bayes Classifiers"""
    
    def __init__(self,tweets):
        self.tweets=tweets
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.tweets['lemmatized_text'],self.tweets['target'],test_size=0.2,stratify=self.tweets['target'])


    def ensemble_Classifier(self):
        # Pre-Trained SVM Classifier
        svm_classifier = joblib.load('Resources/Svm_Classifier_linear.pkl')

        # Vectorizer used for training
        vectorizer = joblib.load('Resources/SVM_Vectorizer.pkl')
        
        # Transforming test data using count vectorizer
        svm_naive_tweets=vectorizer.transform(self.x_test)

        # Predicting test data values
        svm_prediction=svm_classifier.predict(svm_naive_tweets)

        # Pre-Trained Naive Bayes Classifier
        naive_classifier = joblib.load('Resources/NaiveBayes_Classifier.pkl')

        # Predicting test data values
        naive_prediction=naive_classifier.predict(svm_naive_tweets)

        # Pre-Trained LSTM Classifier
        lstm_classifier = load_model('Resources/NeuralNetworkLSTM.h5')

        # LSTM Tokenizer used for training
        tokenizer = joblib.load('Resources/lstm_tokenizer.pkl')
        tokenizer.fit_on_texts(self.tweets['lemmatized_text'].values)
        lstm_tweets = tokenizer.texts_to_sequences(self.x_test)
        lstm_tweets = pad_sequences(lstm_tweets)

        # Preditcing test data values
        lstm_prediction=lstm_classifier.predict_classes(lstm_tweets)
        lstm_prediction = list(itertools.chain(*lstm_prediction))

        # Votting on multiple combinations
        final_pred = np.array([])
        for i in range(len(self.x_test)):
            final_pred = np.append(final_pred, mode([svm_prediction[i], naive_prediction[i], lstm_prediction[i]]))


        report = classification_report(self.y_test, final_pred)
        print("Ensemble")
        print(report)



        

        




