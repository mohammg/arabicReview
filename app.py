# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 14:26:03 2021

@author: moham
"""
import json
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.sav', 'rb'))
labels=pickle.load(open('label.sav','rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    text_sent = [x for x in request.form.values()]
   
    prediction = model.predict(text_sent)

    output = labels.inverse_transform(prediction)

    return render_template('index.html', prediction_text='Review Is  {}'.format(output[0]))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    text_sent = [x for x in data.values()]
    prediction = model.predict(text_sent)
    output = labels.inverse_transform(prediction)
    #output = prediction[0]
    #resultes = [serialize(x) for x in output]
    
    return jsonify(output[0])

def serialize(v):
    return {
        "value" :v,
       
    } 
if __name__ == "__main__":
    app.run(debug=True)
