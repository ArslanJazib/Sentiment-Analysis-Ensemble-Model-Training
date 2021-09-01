import pandas as pd
import time
import joblib
import tweepy
import json
import nltk
import matplotlib.pyplot as plt
import numpy as np
import gensim
import gensim.corpora as corpora
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import API
from tweepy import Stream
from pandas.api.types import CategoricalDtype
from plotnine import *
from wordcloud import WordCloud ,STOPWORDS
from plotnine import ggplot, aes, geom_bar
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import Preprocessor 
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import SVM_Classifier
import Create_Svm_Classifiers





def generate_preprocessed_csv():
    # Loading Sentiment140 Dataset
    tweets = pd.read_csv("Sentiment140.csv" ,names=['target', 'id', 'date','flag','user','text'],encoding='latin-1')

    # Postive = 1 & Negative = 0
    tweets.target = tweets.target.replace({0: 0, 4: 1})

    # Data Preprocessor Object is initlized, Passing dataset into the parameterized constructor
    preprocessor=Preprocessor.Data_Preprocessor(tweets['text'])

    # Fucntion applies all preprocessing functions on tweets and return lemmatized text for each tweet
    tweets['lemmatized_text']=preprocessor.process_tweets()

    tweets.to_csv("Sentiment140_processed.csv")

    # Deleting the tweets dataframe to free the used memeory
    del tweets


def feature_generator(tweets):
    # Create count vectorizer to extract features from text using frequncy of occurance
    vectorizer = CountVectorizer()
    vectorized_data = vectorizer.fit_transform(tweets['lemmatized_text'])
    indexed_data = hstack((np.array(range(0,vectorized_data.shape[0]))[:,None], vectorized_data))
    joblib.dump(indexed_data, 'Sentiment140_features.pkl')
    del vectorizer
    del vectorized_data
    del indexed_data



if __name__ == "__main__":

    # This function is to be called once for the Data Pre-Processing Phase
    #generate_preprocessed_csv()

    # Loading Sentiment140 Dataset
    tweets = pd.read_csv("Sentiment140_processed.csv",encoding='latin-1')
    # drop the first column in dataset
    tweets = tweets.iloc[: , 1:]

    # Droping the Nan rows
    tweets = tweets.dropna(subset=['lemmatized_text'])

   
    # This function will be called once to generate features from the complete dataset
    #feature_generator(tweets)

    features = joblib.load('Sentiment140_features.pkl')
    
    # This function is to be called once for training SVM base classifiers 
    #creator=Create_Svm_Classifiers.Create(target,text)
    #creator.train_svm_models()
    
    #tweets=tweets.sample(n=200000, random_state=1)
    
    # Classifying using support vector machine
    svm_classifier=SVM_Classifier.SupportVectorMachine(features, tweets)
    #svm_classifier.svm_ensembleClassifier()
    svm_classifier.svm_classifiy()