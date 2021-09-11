import pandas as pd
import numpy as np
import re
import nltk
import joblib
from nltk.corpus import stopwords
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import LSTM
from tqdm import tqdm
from time import sleep
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import load_model
from sklearn.metrics import classification_report



class NeuralNetwork_Classifier(object):
    """To create an LSTM Neural Network"""
     
    def __init__(self,tweets):
        self.tweets = tweets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.tweets['lemmatized_text'], self.tweets['target'], test_size=0.20, random_state=42)
        self.embedding_matrix=0
        self.maxlen=0
        self.model=0
        self.vocab_size=0


    def prepare_embedding_layer(self):
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(self.X_train)
        # Saving tokenizer
        joblib.dump(tokenizer, 'Resources/lstm_tokenizer.pkl')
        self.X_train = tokenizer.texts_to_sequences(self.X_train)
        # Adding 1 because of reserved 0 index
        self.vocab_size = len(tokenizer.word_index) + 1
        self.maxlen = 100
        self.X_train = pad_sequences(self.X_train, padding='post', maxlen=self.maxlen)
        embeddings_dictionary = dict()
        glove_file = open('Resources/glove.6B.100d.txt', encoding="utf8")
        for line in glove_file:
            records = line.split()
            word = records[0]
            vector_dimensions = asarray(records[1:], dtype='float32')
            embeddings_dictionary [word] = vector_dimensions
        glove_file.close()
        self.embedding_matrix = zeros((self.vocab_size, 100))
        for word, index in tokenizer.word_index.items():
            embedding_vector = embeddings_dictionary.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[index] = embedding_vector

    def create_architecture(self):
        # Defining the architecture of the neural network
        self.model = Sequential()
        embedding_layer = Embedding(self.vocab_size, 100, weights=[self.embedding_matrix], input_length=self.maxlen , trainable=False)
        self.model.add(embedding_layer)
        self.model.add(LSTM(128))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        print(self.model.summary())

    def lstm_classification(self):
        self.prepare_embedding_layer()
        self.create_architecture()
        self.model.fit(self.X_train, self.y_train, batch_size=128, epochs=1, verbose=1, validation_split=0.2)
        scores = model.evaluate(self.X_test, self.y_test, verbose=1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))
        # save model and architecture to single file
        model.save("Resources/NeuralNetworkLSTM.h5")
        print("Saved model to disk")

    def classification_report(self):
        
        # Pre-Trained LSTM Classifier
        lstm_classifier = load_model('Resources/NeuralNetworkLSTM.h5')

        # LSTM Tokenizer used for training
        tokenizer = joblib.load('Resources/lstm_tokenizer.pkl')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.tweets['lemmatized_text'], self.tweets['target'], test_size=0.20, random_state=42)
        self.X_test = tokenizer.texts_to_sequences(self.X_test)
        pred = pad_sequences(self.X_test, padding='post', maxlen=100)

        # Preditcing test data values
        prediction=lstm_classifier.predict_classes(pred)

        # Classification report
        report = classification_report(self.y_test, prediction)

        print(report)


