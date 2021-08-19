import pandas as pd
import time
import tweepy
import json
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
import SVM_Classifier
import DataSet_insight


def term_documentMatrix(tweets):
    #  limit features to 20 terms
    vect = CountVectorizer(min_df=0., max_df=1.0, max_features=len(tweets))
    Z = vect.fit_transform(tweets)
    tdm=pd.DataFrame(Z.A, columns=vect.get_feature_names())
    tdm=tdm.transpose()
    tdm.index.name = "Terms"
    print(tdm.to_string())
    return tdm

def visualizations(tdm,tweets):
    # Bar PLot
    sum=tdm.sum(axis = 1).reset_index(name ="Count")
    plot=ggplot(sum,aes(x="Terms",y="Count")) + geom_bar(stat="identity") +\
    xlab("Terms") + ylab("Count") + coord_flip()
    print(plot)
    # Word Cloud
    text = tweets
    wordcloud = WordCloud(
        width = 3000,
        height = 2000,
        background_color = 'white',
        stopwords = STOPWORDS
        ).generate(str(text))
    plt.figure(2)
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()

def findAssociations(tdm,word_list,corr_coef):
    similarity_matrix=pd.DataFrame(np.corrcoef(tdm.T),index=tdm.index,columns=tdm.index)
    print(similarity_matrix)
    for value in word_list:
        print(value)
        print(similarity_matrix[similarity_matrix[value]>corr_coef][value].sort_values(ascending=False),"\n")


tweet_list=[]
class SListener(StreamListener):
    def __init__(self,api=None):
        super(SListener,self).__init__()
        self.num_tweets=0
        self.file=open("tweet.txt","w")
    def on_status (self,status):
        tweet=status._json
        # All in lowercase
        tweet['text']=tweet['text'].lower()
        # Remove emoji
        tweet['text']=removeEmojify(tweet['text'])
        # Remove URL
        tweet['text']=removeUrl(tweet['text'])
        # Remove Stop words
        tweet['text']=removeStopwords(tweet['text'])
        # Stemming Tweets
        # For displaying differnce between Stemming and Lemmatization
        temp=tweet['text']
        tweet['text']=stemmer(tweet['text'])
        # Lemmatizing Tweets
        temp=lemmatizer(temp)
        self.file.write(json.dumps(tweet)+ '\n')
        tweet_list.append(status)
        self.num_tweets+=1
        if self.num_tweets<20:
            return True
        else:
            return False
        self.file.close()


def flatten_tweets():
    """ Flattens out tweet dictionaries so relevant JSON
        is in a top-level dictionary."""
    tweets_list = []
   
    # Iterate through each tweet
    for tweet in open('tweet.txt','r'):
        tweet_obj = json.loads(tweet)
        # Store the user screen name in 'user-screen_name'
        tweet_obj['user-screen_name'] = tweet_obj['user']['screen_name']
    
        # Check if this is a 140+ character tweet
        if 'extended_tweet' in tweet_obj:
            # Store the extended tweet text in 'extended_tweet-full_text'
            tweet_obj['extended_tweet-full_text'] = tweet_obj['extended_tweet']['full_text']
    
        if 'retweeted_status' in tweet_obj:
            # Store the retweet user screen name in 'retweeted_status-user-screen_name'
            tweet_obj['retweeted_status-user-screen_name'] = tweet_obj['retweeted_status']['user']['screen_name']

            # Store the retweet text in 'retweeted_status-text'
            tweet_obj['retweeted_status-text'] = tweet_obj['retweeted_status']['text']
            
        tweets_list.append(tweet_obj)
    return tweets_list







if __name__ == "__main__":
    # Loading Sentiment140 Dataset
    tweets = pd.read_csv("Sentiment140.csv" ,names=['target', 'id', 'date','flag','user','text'],encoding='latin-1')
    # Postive = 1 & Negative = 0
    tweets.target = tweets.target.replace({0: 0, 4: 1})
    # Making a smaller sample of the data
    tweets = tweets.sample(n=100000)
    # Data Preprocessor Object is initlized, Passing dataset into the parameterized constructor
    preprocessor=Preprocessor.Data_Preprocessor(tweets['text'])
    # Fucntion applies all preprocessing functions on tweets and return lemmatized text for each tweet
    tweets['lemmatized_text']=preprocessor.process_tweets()
    svm_classifier=SVM_Classifier.SupportVectorMachine(tweets)
    svm_classifier.svm_classifiy()

    # Insight object being initialzed for geneting insights about the dataset being used
    #report=DataSet_insight.Reporter(tweets)
    # Will display information on data set
    #report.report_generator()


    
    




