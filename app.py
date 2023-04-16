import numpy as np
from flask import Flask, request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [[x for x in request.form.values()]]
    prediction = model.predict(int_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Quantitative measure of disease progression one year after baseline: {:.2f}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)