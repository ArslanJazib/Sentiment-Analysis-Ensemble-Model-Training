import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report



class NeuralNetwork_Classifier(object):
    """To create an LSTM Neural Network"""
     
    def __init__(self,features,tweets):
        self.tweets = tweets
        self.features = features
        self.model=0

    def create_architecture(self):
        # Defining the architecture of the neural network
        embed_dim = 128
        lstm_out = 196
        self.model = Sequential()
        self.model.add(Embedding(2000, embed_dim,input_length = self.features.shape[1]))
        self.model.add(SpatialDropout1D(0.4))
        self.model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
        print(self.model.summary())

    def lstm_classification(self):
        self.create_architecture()
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.tweets['target'], test_size=0.20, random_state=42)
        self.model.fit(X_train, y_train, epochs = 10, batch_size=128, verbose = 1)
        scores = self.model.evaluate(X_test, y_test, verbose=1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))
        # save model and architecture to single file
        model.save("NeuralNetworkLSTM.h5")
        print("Saved model to disk")


    def classification_report(self):
        
        # Pre-Trained LSTM Classifier
        lstm_classifier = load_model('Resources/NeuralNetworkLSTM.h5')

        # LSTM Tokenizer used for training
        tokenizer = load_model('Resources/LSTM_Tokenizer')
        tokenizer.fit_on_texts(self.tweets['lemmatized_text'].values)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.tweets['lemmatized_text'], self.tweets['target'], test_size=0.20, random_state=42)
        self.X_test = tokenizer.texts_to_sequences(self.X_test)
        self.X_test = pad_sequences(self.X_test)

        # Preditcing test data values
        prediction=lstm_classifier.predict_classes(self.X_test)

        # Classification report
        report = classification_report(self.y_test, prediction)

        print(report)



