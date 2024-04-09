from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
cors = CORS(app)
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
machine_dataset = pd.read_csv('machine_failure.csv')

@app.route('/')
def index():
    return render_template('index.html')

# ... your other Flask code ...

@app.route('/predict', methods=['POST','GET'])
@cross_origin()
def predict():
    try:
        # Extract features from form data
        int_features = [float(request.form.get(feature)) for feature in ['Air_temperature_k', 'Process_temperature_k', 'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']]
        
        # Make prediction
        features = [np.array(int_features)]
        prediction = model.predict(features)
        output = round(prediction[0], 2)

        # Return the prediction result
        return render_template('index.html', prediction_text='The percent is {}'.format(output))

    except Exception as e:
        return render_template('index.html', prediction_text='Error: {}'.format(str(e)))

# ... rest of your Flask code ...

