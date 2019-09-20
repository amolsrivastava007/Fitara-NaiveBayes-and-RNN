from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
import numpy as np
import pickle as p
import json
#from tensorflow import keras

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index1.html')



@app.route('/results',methods=['POST'])

def predictions():

    data = request.form['text']

    data = [data]

    data = vect.transform(data)
    
    prediction = np.array2string(model.predict_proba(data))

    return jsonify(prediction)

if __name__ == '__main__':

    modelfile = 'nb.pickle'

    model = p.load(open(modelfile,'rb'))

    vect = p.load(open('cv.pickle','rb'))

    app.run(debug=True)
    
