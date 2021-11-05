import time
import json
import tweepy
import numpy as np
import pandas as pd
from tweepy import API
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener


tweet_list=[]
class SListener(StreamListener):
    def __init__(self,api=None):
        super(SListener,self).__init__()
        self.num_tweets=0
        self.file=open("tweet.txt","w")
    
    def on_status (self,status):
        tweet=status._json
        # All in lowercase
        #tweet['text']=tweet['text'].lower()
        # Remove emoji
        #tweet['text']=removeEmojify(tweet['text'])
        # Remove URL
        #tweet['text']=removeUrl(tweet['text'])
        # Remove Stop words
        #tweet['text']=removeStopwords(tweet['text'])
        # Stemming Tweets
        # For displaying differnce between Stemming and Lemmatization
        #temp=tweet['text']
        #tweet['text']=stemmer(tweet['text'])
        # Lemmatizing Tweets
        #temp=lemmatizer(temp)
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
    #tweets_list = []
   
    # Iterate through each tweet
    #for tweet in open('tweet.txt','r'):
        #tweet_obj = json.loads(tweet)
        # Store the user screen name in 'user-screen_name'
        #tweet_obj['user-screen_name'] = tweet_obj['user']['screen_name']
    
        # Check if this is a 140+ character tweet
        #if 'extended_tweet' in tweet_obj:
            # Store the extended tweet text in 'extended_tweet-full_text'
            #tweet_obj['extended_tweet-full_text'] = tweet_obj['extended_tweet']['full_text']
    
        #if 'retweeted_status' in tweet_obj:
            # Store the retweet user screen name in 'retweeted_status-user-screen_name'
            #tweet_obj['retweeted_status-user-screen_name'] = tweet_obj['retweeted_status']['user']['screen_name']

            # Store the retweet text in 'retweeted_status-text'
            #tweet_obj['retweeted_status-text'] = tweet_obj['retweeted_status']['text']
            
        #tweets_list.append(tweet_obj)
    #return tweets_list

class Api_Authentication(object):
    """"""
    #consumer_key="hqdSn31cvUUMEZZwqJreGDsaZ"
    #consumer_secret="ZWJEHxuJB9DKw4HEDTZ4tMM0h4BpFTEXMNZBb3aIsvt8aMbxJP"
    #access_token="1089143318380986373-fxmqHcDgZJ0GUNLiwCslHJckcU55VR"
    #access_token_secret="ws7iluXMr15eQykR1ur771v87K2p4mPbxzhMutVr6AI73"
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
    # Counting the data collected using the API
    #print("Total tweets collected",len(ds_tweets.index))
    #print("-----------------------Execution Time-----------------------")
    #print("--- %s seconds ---" % (time.time() - start_time))


