from flask import Flask, render_template, request
import pickle
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)

# Load trained model
with open('model/medical_cost_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    sex = 1 if request.form.get('sex') == 'male' else 0
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = 1 if request.form.get('smoker') == 'yes' else 0
    region = request.form.get('region')

    # Create input feature array
    input_data = {
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region_northwest': 0,
        'region_southeast': 0,
        'region_southwest': 0
    }

    if region != 'northeast':  # 'northeast' is dropped in get_dummies(drop_first=True)
        input_data[f'region_{region}'] = 1

    # Arrange feature order to match model
    final_input = np.array([input_data[col] for col in model_columns]).reshape(1, -1)

    prediction = model.predict(final_input)[0]

    return render_template('result.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
