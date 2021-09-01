import pandas as pd
import re, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from textblob import TextBlob

class Data_Preprocessor:
    """This class will pre process the data for the model"""

    def __init__(self, tweets):
        self.tweets = tweets

    def process_tweets(self):
        tweet_list=[]
        for tweet in self.tweets:
            tweet_list.append(self.normalizer(tweet))
        return tweet_list

    def remove_emojis(self,tweet):
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
        return regrex_pattern.sub(r'',tweet)

    def remove_url(self,tweet):
        tweet = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', tweet, flags=re.MULTILINE) 
        return tweet

    def normalizer(self,tweet):
        # To remove emojis from tweet
        tweet=self.remove_emojis(tweet)
        # To remove URL's from tweet
        tweet=self.remove_url(tweet)
        # To remove stop words from tweet
        stop_words = set(stopwords.words('english'))
        # To create lemmas from tweet
        wordnet_lemmatizer = WordNetLemmatizer()
        # Removing punctuations
        only_letters = re.sub("[^a-zA-Z]", " ",tweet)
        # Removing tags @
        tweet_list = only_letters.split()
        index = 0
        while index < len(tweet_list):
            if tweet_list[index][0] == '@':
                tweet_list.remove(tweet_list[index])
                index = 0
            else:
                index+=1
        only_letters =" ".join([str(element) for element in tweet_list])
        # Creating tokens from tweet
        tokens = nltk.word_tokenize(only_letters)[2:]
        # Lower case each letter within the tweet
        lower_case = [l.lower() for l in tokens]
        # Applying the filters created above on tweet
        filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
        # Lemmas are stored in a list
        lemmas = [wordnet_lemmatizer.lemmatize(t,pos="v") for t in filtered_result]
        # Lemmatized text is joined into a sentence which is to be used for model training
        lemmas=" ".join(lemmas)    
        return lemmas




