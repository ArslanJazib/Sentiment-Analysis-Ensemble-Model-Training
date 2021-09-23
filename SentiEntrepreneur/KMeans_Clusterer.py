import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer





class KmeansClusterer(object):
    """To create a k means cluster """

    def __init__(self,features,tweets):
        self.tweets = tweets
        self.features=features


    def classification_report(self):

        kmeans_two_clusters = joblib.load('Resources/KMeans_clusters.pkl')

        # Vectorizer used for training
        vectorizer = joblib.load('Resources/KMeans_Vectorizer.pkl')
        
        # Tranforming test data using count vectorizer
        x_train,x_test,y_train,y_test=train_test_split(self.features,self.tweets['target'],test_size=0.2,stratify=self.tweets['target'])

        # Preditcing test data values
        prediction = kmeans_two_clusters.predict(x_test)

        unique, counts = np.unique(kmeans_two_clusters.labels_, return_counts=True)
        print(dict(zip(unique, counts)))

        # Classification report
        report = classification_report(y_test,prediction)

        print(report)


    def kmeans_cluster(self):
        kmeans_model = KMeans(n_clusters=2,verbose=True)
        x_train,x_test,y_train,y_test=train_test_split(self.features,self.tweets['target'],test_size=0.2,stratify=self.tweets['target'])
        kmeans_two_clusters = kmeans_model.fit(x_train)
        # Storing the model in a file
        joblib.dump(kmeans_two_clusters, 'Resources/KMeans_clusters.pkl')

