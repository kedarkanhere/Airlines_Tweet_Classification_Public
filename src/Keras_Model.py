# -*- coding: utf-8 -*-
#importing libraries
import numpy as np 
import pandas as pd
import re
import gc
import os
import string
import tensorflow as tf
import sys
from tqdm  import tqdm
tqdm.pandas()
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,f1_score, roc_auc_score
import keras
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score,KFold,train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from sklearn.metrics import classification_report
import spacy
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

nlp = spacy.load("en_core_web_sm")
prefixes = ('\\n', ) + nlp.Defaults.prefixes
stops = nlp.Defaults.stop_words

df=pd.read_csv('../input/Usecase3_Dataset.csv')

#Create a function to cleanse the tweet for things like special characters, hashtags, urlss,@ signs.
#Then we will use Spacy NLP to bring the words back to their root form

def normalize(comment, lowercase, remove_stopwords):
    if lowercase:
        comment = comment.lower()
        comment= re.sub(r'@\w+',' ',comment)
        comment=re.sub(r'#',' ',comment)
        comment = re.sub('http\S+',' ',comment)
        comment = re.sub(r"[^A-Za-z0-9']+",' ',comment)
        comment = re.sub(r"\s+",' ',comment)
        

        
    comment = nlp(comment)
    lemmatized = list()
    for word in comment:
        lemma = word.lemma_.strip()
        if lemma:
            if not remove_stopwords or (remove_stopwords and lemma not in stops and lemma != '-PRON-'):
                lemmatized.append(lemma)
    return " ".join(lemmatized)

df['Clean_Text']=df.text.apply(normalize,lowercase=True, remove_stopwords=True)

df['Clean_Text'].head(25)

#Create dummy variables for multiclass classification
Y = pd.get_dummies(df['airline_sentiment']).values
X=df["Clean_Text"].values

#Train test split
X_train, X_test, y_train, y_test = train_test_split(df["Clean_Text"].values,Y, test_size=0.2, random_state=87,stratify=Y)

#Initialize the vocab size and fit text through tokenizer
vocabulary_size = 4000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(X_train)
sequences = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(sequences, maxlen=100)

sequences = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(sequences, maxlen=100)

#Clear backend keras session if any
keras.backend.clear_session()

#Start the sequential model
model = Sequential()
model.add(Embedding(vocabulary_size, 10, input_length=100))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.4))
model.add(LSTM(100))
#model.add(Flatten())
model.add(Dense(3, activation='softmax'))

#print model summary
model.summary()

#Using categorical crossentropy and Nadam (Adam also gives similar results)
model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
#Defining early stopping rounds. We dont wish to overfit
stopping_rounds=EarlyStopping(monitor='val_loss', mode='max', verbose=1, patience=2)

#Fitting the model
hist=model.fit(X_train, y_train,
                    batch_size=128,
                    epochs=8,
                    verbose=1,
                    validation_split=0.1,
                    callbacks=[stopping_rounds],
                    shuffle=True
          )

#Predictions
preds = model.predict(X_test)

#Looking at PR & Accuracy, F1 score
print(classification_report(np.argmax(y_test,axis=1),np.argmax(preds,axis=1)))

#PLotting Validation vs Training history
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']

# Get the amount of epochs for visualization
stopped_epoch = stopping_rounds.stopped_epoch
n_epochs = range(stopped_epoch+1)

# Plot training and validation accuracy
plt.figure(figsize=(15,5))
plt.plot(n_epochs, acc)
plt.plot(n_epochs, val_acc)
plt.title('Accuracy over epochs', weight='bold', fontsize=22)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.legend(['Training accuracy', 'Validation accuracy'], fontsize=16)
plt.show()

