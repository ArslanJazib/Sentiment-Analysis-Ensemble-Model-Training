import pandas as pd
import time
import joblib
import tweepy
import json
import nltk
import os
import matplotlib.pyplot as plt
import numpy as np
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import API
from tweepy import Stream
from pandas.api.types import CategoricalDtype
import Preprocessor 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import classification_report
from scipy.sparse import hstack
import SVM_Classifier
import NeuralNetwork_Classifier as neural
import KMeans_Clusterer as kmeans
import NaiveBayes_Classifier as naive





def generate_preprocessed_csv():
    # Loading Sentiment140 Dataset
    tweets = pd.read_csv("Resources/Sentiment140.csv" ,names=['target', 'id', 'date','flag','user','text'],encoding='latin-1')

    # Postive = 1 & Negative = 0
    tweets.target = tweets.target.replace({0: 0, 4: 1})

    # Data Preprocessor Object is initlized, Passing dataset into the parameterized constructor
    preprocessor=Preprocessor.Data_Preprocessor(tweets['text'])

    # Fucntion applies all preprocessing functions on tweets and return lemmatized text for each tweet
    tweets['lemmatized_text']=preprocessor.process_tweets()

    # To remove the extra 0th column of indexes
    tweets = tweets.iloc[: , 1:]

    # Droping the Nan rows
    tweets = tweets.dropna(subset=['lemmatized_text'])

    # Creating a temporary dataframe to hold the suffled data
    shuffled_Tweets = pd.DataFrame(0, index=np.arange(1546126), columns=['target', 'id', 'date','flag','user','text','lemmatized_text'])

    # Negative values from the 0th index till 773062 are placed at even index 
    shuffled_Tweets[0:1546124:2]=tweets[0:773062]
    
    # Positive values from the 780390 index till 1553452 are placed at even index 
    shuffled_Tweets[1:1546125:2]=tweets[780390:1553452]
    
    # Remaing extra negative tweets from index 773063 to 780380 are appended at the end of the dataframe
    shuffled_Tweets=pd.concat([shuffled_Tweets, tweets[773063:780389]])

    shuffled_Tweets.to_csv("Resources/Sentiment140_processed.csv")

    # Deleting the variables to free used memeory
    del tweets
    del shuffled_Tweets
    del preprocessor


def feature_generator(tweets):
    # Create count vectorizer to extract features from text using frequncy of occurance
    # For SVM
    vectorizer = CountVectorizer()
    joblib.dump(vectorizer, 'Resources/SVM_Vectorizer.pkl')
    vectorized_data = vectorizer.fit_transform(tweets['lemmatized_text'])
    indexed_data = hstack((np.array(range(0,vectorized_data.shape[0]))[:,None], vectorized_data))
    joblib.dump(indexed_data, 'Resources/Sentiment140_SVM_features.pkl')
    # For KMeans
    vectorizer = TfidfVectorizer(max_features=200000, stop_words='english',use_idf=True, ngram_range=(1,3))
    joblib.dump(vectorizer, 'Resources/KMeans_Vectorizer.pkl')
    indexed_data = vectorizer.fit_transform(tweets['lemmatized_text'])
    joblib.dump(indexed_data, 'Resources/Sentiment140_KMeans_features.pkl')
    # For Neural Network
    tokenizer = Tokenizer(num_words=2000, split=' ')
    tokenizer.fit_on_texts(tweets['lemmatized_text'].values)
    joblib.dump(tokenizer, 'Resources/LSTM_Tokenizer.pkl')
    data = tokenizer.texts_to_sequences(tweets['lemmatized_text'].values)
    data = pad_sequences(data)
    joblib.dump(data, 'Resources/Sentiment140_LSTM_features.pkl')
    del vectorizer
    del vectorized_data
    del indexed_data
    del tokenizer
    del data


def model_training(tweets):

    # Choose a model to be trained
    choice = input(" Press 1 for training SVM Classifier \n Press 2 for training LSTM Neural Network Classifier \n Press 3 for training Naive Bayes Classifier \n Press 4 for training using KMeans Clustering \n Enter Choice: ")
    
    if choice=='1':    
        
        features = joblib.load('Resources/Sentiment140_SVM_features.pkl')

        # Classifying using support vector machine
        svm_classifier=SVM_Classifier.SupportVectorMachine(features, tweets)
        
        # Model Training SVM
        svm_classifier.svm_ensembleClassifier()
        

    elif choice=='2':    

        features = load_model('Resources/Sentiment140_LSTM_features')

        # Classifying using Recurrent Neural Network (LSTM)
        neural_classifier=neural.NeuralNetwork_Classifier(features,tweets)
        
        # Model Training LSTM
        neural_classifier.lstm_classification()
        

    elif choice=='3':   
        
        features = joblib.load('Resources/Sentiment140_SVM_features.pkl')

        # Classifying using naive bayes classifier
        naive_model=naive.NaiveBayes_Classifier(features, tweets)
        
        # Model Training naive bayes
        naive_model.naive_Classifier()
        

    elif choice=='4':
        
        features = joblib.load('Resources/Sentiment140_KMeans_features.pkl')

        # Clustering using KMeans
        kmeans_cluster=kmeans.KmeansClusterer(features, tweets)
    
        # Model Training KMeans
        kmeans_cluster.kmeans_cluster()


def classification_reports(tweets):
    # Choose to get a classification report on a the trained model
    choice = input(" Press 1 for testing SVM Classifier \n Press 2 for testing LSTM Neural Network Classifier \n Press 3 for testing Naive Bayes Classifier \n Press 4 for testing using KMeans Clustering \n Enter Choice: ")
    
    if choice=='1':
        
        features = joblib.load('Resources/Sentiment140_SVM_features.pkl')

        # Classifying using support vector machine
        svm_classifier=SVM_Classifier.SupportVectorMachine(features, tweets)
        
        # Model Testing SVM
        svm_classifier.classification_report()
        

    elif choice=='2':    
        # Classifying using Recurrent Neural Network (LSTM)
        neural_classifier=neural.NeuralNetwork_Classifier(tweets)
        
        # Model Testing LSTM
        neural_classifier.classification_report()
        

    elif choice=='3':    

        features = joblib.load('Resources/Sentiment140_SVM_features.pkl')

        # Classifying using naive bayes classifier
        naive_model=naive.NaiveBayes_Classifier(features, tweets)
        
        # Model Testing naive bayes
        naive_model.classification_report()
        

    elif choice=='4':    

        features = joblib.load('Resources/Sentiment140_KMeans_features.pkl')

        # Clustering using KMeans
        kmeans_cluster=kmeans.KmeansClusterer(features, tweets)
    
        # Model Testing KMeans
        kmeans_cluster.classification_report()

if __name__ == "__main__":

    # This function is to be called once for the Data Pre-Processing Phase
    #generate_preprocessed_csv()

    # Loading Sentiment140 Dataset
    tweets = pd.read_csv("Resources/Sentiment140_processed.csv",encoding='latin-1')

    # This function will be called once to generate features from the complete dataset
    feature_generator(tweets)

    # Taking input from user to train a model
    choice = input(" Press 1 for Model Training  \n Press 2 for Model Testing \n Enter Choice: ")
    
    if choice=='1':    
        os.system('cls')
        model_training(tweets)

    elif choice=='2':  
        os.system('cls')
        classification_reports(tweets)
