import collections
from nltk import ngrams

class Reporter:
    """This class is to help get an assememnt on the data beofre training a model"""

    def __init__(self, tweets,lemmas):
        self.tweets = tweets
        self.lemmas=lemmas

    def report_generator(self):
      # To check the class imbalance in the target attribute
      print("Postive & Negative distrubtion in the Sentiment140 Dataset\n")
      self.check_imbalance(self.tweets)
      # To generate n-grams from the lemmas
      self.tweets['grams']=self.ngrams_generator()
      # Printing most frequent n-grams for positive and negative labels
      print("20 of the most frequetnly occured n-grams having positive label\n")
      self.tweets[(tweets['target'] == 4)][['grams']].apply(self.count_words)['grams'].most_common(20)
      print("20 of the most frequetnly occured n-grams having negative label\n")
      self.tweets[(tweets['target'] == 0)][['grams']].apply(self.count_words)['grams'].most_common(20)

    def check_imbalance(self,tweets):
        print(tweets['target'].value_counts(normalize=True))
    
    # To generate n-grams from 
    def ngrams_generator(self,leemas):
        #onegrams = input_list
        bigrams = [' '.join(t) for t in list(zip(lemmas, lemmas[1:]))]
        trigrams = [' '.join(t) for t in list(zip(lemmas, lemmas[1:], lemmas[2:]))]
        return bigrams+trigrams





