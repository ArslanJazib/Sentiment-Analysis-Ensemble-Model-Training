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

class Api_Authentication(object):
    """description of class"""
    #start_time = time.time()
    #consumer_key=" "
    #consumer_secret=""
    #access_token=" "
    #access_token_secret=""
    #auth = OAuthHandler(consumer_key,consumer_secret)
    #auth.set_access_token(access_token,access_token_secret)
    #listen = SListener()
    #stream = Stream(auth,listen)
    #stream.filter(track=['Python'],languages=["en"]);
    #tweets = flatten_tweets()
    # Create a DataFrame from `tweets`
    #ds_tweets = pd.DataFrame(tweets)
    # Print out tweets from this dataset
    #print(ds_tweets['text'].values)
    # Term Document Matrix
    #tdm=term_documentMatrix(ds_tweets['text'].values)
    # Finding associations
    #findAssociations(tdm,[tdm.index[0],tdm.index[1]],0.8)
    # Word Cloud and Bar Chart
    #visualizations(tdm,ds_tweets['text'].values)    
    # Counting the data collected using the API
    #print("Total tweets collected",len(ds_tweets.index))
    #print("-----------------------Execution Time-----------------------")
    #print("--- %s seconds ---" % (time.time() - start_time))


