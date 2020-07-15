#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:58:53 2020

@author: kush
"""

from flask import Flask,render_template,request
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
IMAGE_FOLDER = os.path.join('static', 'img_pool')

rbf = pickle.load(open('rbf_model.pkl','rb'))
cv = pickle.load(open('cv_transform.pkl','rb'))

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/emotion_predict',methods=['POST', "GET"])
def emotion_predict():
    if request.method =='POST':
        text = request.form['text']
        text = [text]
        cvect = cv.transform(text).toarray()
        emotion_pred = rbf.predict(cvect)
        probability = rbf.predict_proba(cvect)
        probability = probability.max()
        if emotion_pred == 'anger':
            emotion = 'Anger'
            image = os.path.join(app.config['UPLOAD_FOLDER'], 'anger.png')
        elif emotion_pred == 'disgust':
            emotion = 'Disgust'
            image = os.path.join(app.config['UPLOAD_FOLDER'], 'disgust.png')
        elif emotion_pred =='fear':
            emotion = 'Fear'
            image = os.path.join(app.config['UPLOAD_FOLDER'], 'fear.png')
        elif emotion_pred == 'guilt':
            emotion  = 'Guilt'
            image = os.path.join(app.config['UPLOAD_FOLDER'], 'guilt.png')
        elif emotion_pred == 'joy':
            emotion = 'Joy'
            image = os.path.join(app.config['UPLOAD_FOLDER'], 'joy.png')
        elif emotion_pred == 'sadness':
            emotion  ='Sadness'
            image = os.path.join(app.config['UPLOAD_FOLDER'], 'sadness.png')
        else:
            emotion = 'Shame'
            image = os.path.join(app.config['UPLOAD_FOLDER'], 'shame.png')
    return render_template('home.html', text=text, emotion=emotion, probability=probability,image=image)

if __name__ == '__main__':
    app.run()
    
