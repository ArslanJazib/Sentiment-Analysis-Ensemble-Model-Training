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
from sklearn.feature_extraction.text import CountVectorizer
from pandas.api.types import CategoricalDtype
from plotnine import *
from wordcloud import WordCloud ,STOPWORDS
from plotnine import ggplot, aes, geom_bar
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import Preprocessor 


def removeEmojify(text):
    regrex_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u'\U00010000-\U0010ffff'
                u"\u200d"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\u3030"
                u"\ufe0f"
    "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'',text)

def removeUrl(text):
    text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', text, flags=re.MULTILINE) 
    return text

def removeStopwords(text):
    stop_words = set(stopwords.words('english')) 
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stop_words]
    filtered_sentence = (" ").join(tokens_without_sw)
    return filtered_sentence

def stemmer(text):
    #create an object of class PorterStemmer
    porter = PorterStemmer()
    token_words=word_tokenize(text)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    #print("{0:20}{1:20}".format("Word","Porter Stemmer"))
    #for word in token_words:
    #    print("{0:20}{1:20}".format(word,porter.stem(word)))
    return "".join(stem_sentence)

def lemmatizer(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    sentence = text
    punctuations="?:!.,;"
    sentence_words = nltk.word_tokenize(sentence)
    leema_sentence=[]
    for word in sentence_words:
        if word in punctuations:
            sentence_words.remove(word)
        else:
            leema_sentence.append(wordnet_lemmatizer.lemmatize(word,pos="v"))
            leema_sentence.append(" ")
    #print("{0:20}{1:20}".format("Word","Lemma"))
    #for word in sentence_words:
    #    print ("{0:20}{1:20}".format(word,wordnet_lemmatizer.lemmatize(word,pos="v")))
    return "".join(leema_sentence)

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
    tweets = pd.read_csv("Sentiment140.csv" ,names=['target', 'id', 'date','flag','user','text'])
    # Data Preprocessor Object is initlized, Passing dataset into the parameterized constructor
    preprocessor=Preprocessor.Data_Preprocessor(tweets['text'])
    # Fucntion applies all preprocessing functions on tweets and returns a list of lemmas for each tweet
    leemas=preprocessor.process_tweets()
    print(leemas)

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

