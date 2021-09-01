import joblib
import SVM_Classifier
import pandas as pd
from sklearn import svm
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer

class Create(object):
    """This class is used in making smaller samples of the dataset to train mutiple models because of hardware restrictions"""
    
    def __init__(self, features, tweets):
        self.tweets = tweets
        self.features=features



    def svm_classifiy(self,tweets,file_counter):
        vector=self.feature_generator(tweets,file_counter)
        # Perform classification with SVM, kernel=linear
        svm_classifier_linear = svm.SVC(kernel='linear')
        svm_classifier_linear.fit(vector,tweets['target'])
        return svm_classifier_linear

    def train_svm_models(self):

        # Making smaller samples of the dataset to train mutiple models because of hardware restrictions
        
        #sample1 = [0,99999,800000,899999]
        sample2 = [100000,199999,900000,999999]
        sample3 = [200000,299999,1000000,1099999]
        sample4 = [300000,399999,1100000,1199999]
        sample5 = [400000,499999,1200000,1299999]
        sample6 = [500000,599999,1300000,1399999]
        sample7 = [600000,699999,1400000,1499999]
        sample8 = [700000,799999,1500000,1553452]

        #samples=[sample1,sample2,sample3,sample4,sample5,sample6,sample7,sample8]
        samples=[sample2,sample3,sample4,sample5,sample6,sample7,sample8]
        file_counter=1
        # Samples
        for indexes in samples:
            tweets_pos = self.tweets.iloc[indexes[0]:indexes[1]+1]
            tweets_neg = self.tweets.iloc[indexes[2]:indexes[3]+1]
            # Concatenating positive & negative samples
            frames = [tweets_pos, tweets_neg]
            tweets = pd.concat(frames)
            # Traning SVM Base Model on the training data
            svm_classifier_linear=self.svm_classifiy(tweets,file_counter)
            # Save the model as a pickle in a file
            fileName='SVM_Linear_Classifier'+str(file_counter)+'.pkl'
            joblib.dump(svm_classifier_linear, fileName)
            # Deleting used local variables from the memory
            file_counter=file_counter+1
            del tweets_pos
            del tweets_neg
            del frames
            del svm_classifier_linear