# https://youtu.be/bluclMxiUkA
"""
Application that predicts heart disease percentage in the population of a town
based on the number of bikers and smokers. 

Trained on the data set of percentage of people biking 
to work each day, the percentage of people smoking, and the percentage of 
people with heart disease in an imaginary sample of 500 towns.

"""


import numpy as np
from flask import Flask, request, render_template
import pickle

#Create an app object using the Flask class. 
app = Flask(__name__)
#app = Flask(__name__, '/static')

#Load the trained model. (Pickle file)
model = pickle.load(open('training_model/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/trynow')
def try_now():
    return render_template('trynow.html')
@app.route('/contactt')
def contactt():
    return render_template('contactt.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()] #Convert string inputs to float.
    features = [np.array(int_features)]  #Convert to the form [[a, b]] for input to the model
    prediction = model.predict(features)  # features Must be in the form [[a, b]]

    output = ((prediction[0]))

    return render_template('contactt.html', prediction_text='you have lung cancer {}'.format(output))


if __name__ == "__main__":
    app.run(host='localhost', port=3900)