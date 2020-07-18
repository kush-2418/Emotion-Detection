#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:58:53 2020

@author: kush
"""

from flask import Flask, render_template, flash, request, url_for, redirect, session
import os
import pickle


rbf = pickle.load(open('rbf_model.pkl','rb'))
cv = pickle.load(open('cv_transform.pkl','rb'))

app = Flask(__name__)


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
            emotion = 'Anger 😡'
            
        elif emotion_pred == 'disgust':
            emotion = 'Disgust 😖'
        elif emotion_pred =='fear':
            emotion = 'Fear 😨'
        elif emotion_pred == 'guilt':
            emotion  = 'Guilt 🙄'
        elif emotion_pred == 'joy':
            emotion = 'Joy 😃'
        elif emotion_pred == 'sadness':
            emotion  ='Sadness 😥'
        else:
            emotion = 'Shame 🙁'
    return render_template('home.html', text=text, emotion=emotion, probability=probability)

if __name__ == '__main__':
    app.run()
    
