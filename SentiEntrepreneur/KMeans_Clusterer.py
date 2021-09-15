import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

class KmeansClusterer(object):
    """To create a k means cluster """

    def __init__(self, features, tweets):
        self.tweets = tweets
        self.features=features
        # Splitting each sample data set into training and testing sets with the ratio of 80:20
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.features,self.tweets['target'],test_size=0.2,stratify=tweets['target'])
        self.x_train = self.x_train[:,1:]
        self.x_test = self.x_test[:,1:]


    def classification_report(self):
        
        kmeans_two_clusters = joblib.load('Resources/KMeans_clusters.pkl')

        # Vectorizer used for training
        vectorizer = joblib.load('Resources/Vectorizer.pkl')
        
        # Tranforming test data using count vectorizer
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.tweets['lemmatized_text'],kmeans_two_clusters.labels_,test_size=0.2,stratify=kmeans_two_clusters.labels_)
        self.x_test=vectorizer.transform(self.x_test)

        # Preditcing test data values
        prediction = kmeans_two_clusters.predict()

        #unique, counts = np.unique(kmeans_two_clusters.labels_, return_counts=True)
        #print(dict(zip(unique, counts)))

        # Classification report
        report = classification_report(self.y_test,prediction)

        print(report)


    def kmeans_cluster(self):
        
       kmeans_two_clusters = KMeans( n_clusters=2, n_init=1000000, random_state=1,verbose=1)

       kmeans_two_clusters.fit(self.x_train)

       # Storing the model in a file
       joblib.dump(kmeans_two_clusters, 'Resources/KMeans_clusters.pkl')