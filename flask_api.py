# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 17:15:02 2021

@author: tg248
"""

from flask import Flask,request, render_template
import pandas as pd
import numpy as np
import pickle


app= Flask(__name__)
pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def home():
    variance=request.form['variance']
    skewness=request.form['skewness']
    curtosis=request.form['curtosis']
    entropy=request.form['entropy']
    arr=np.array([[variance,skewness,curtosis,entropy]])
    prediction=classifier.predict(arr)
    return render_template('index.html',data=prediction)



@app.route('/predict_file', methods=["POST"])
def predict_note_authentication1():
    df_test=pd.read_csv(request.files.get("file"))
    prediction= classifier.predict(df_test)
    return "The predicted values for csv are" +str(list(prediction))




if __name__=='__main__':
    app.run()