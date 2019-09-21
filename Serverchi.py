from flask import Flask, request, redirect, url_for, flash, jsonify,render_template
import numpy as np
import pickle as p
import json
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
from keras.callbacks import EarlyStopping
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU, Flatten, SpatialDropout1D

app = Flask(__name__)


@app.route('/')

def home():
    return render_template('home.html')



@app.route('/results',methods=['POST'])
def predict():
    print("Hello")
    # The maximum number of words to be used. (most frequent)
    MAX_NB_WORDS = 50000
    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = 250
    # This is fixed.
    EMBEDDING_DIM = 100
    print("Hello2")
    new_model=tf.keras.models.load_model('mymodel.h5')
    print(new_model.summary())
    if request.method=='POST':
     comment=request.form['comment']
    print(comment)
    print("Outside")

    data=pd.Series(comment)


    data = data.str.lower()
    data = data.str.replace('[^\w\s]','')
    data = data.fillna("fillna")
    data = data.str.replace('\d+', '')
    #data = data.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    #data = vect.transform(data)

   
    
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(data.values)
    word_index = tokenizer.word_index

    X = tokenizer.texts_to_sequences(data.values)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    print(new_model.summary())
    prediction=new_model.predict(X)
    
    
    prediction=prediction[0][1]
    print(prediction)
    #Pass to Prediction folder
    return render_template('result.html', prediction =prediction)

    

if __name__ == '__main__':

    
    app.run(debug=True)
