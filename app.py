from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model

# Load the model
model = load_model(r'C:\Users\KIIT\Desktop\My learning\PROJECTS\Heart_disease_prediction\models\logistic_regression_model.keras')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    sex = float(request.form['sex'])
    cp = float(request.form['cp'])
    trestbps = float(request.form['trestbps'])
    chol = float(request.form['chol'])
    fbs = float(request.form['fbs'])
    restecg = float(request.form['restecg'])
    thalach = float(request.form['thalach'])
    exang = float(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = float(request.form['slope'])
    ca = float(request.form['ca'])
    thal = float(request.form['thal'])

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)

    # Print the prediction value to the console
    print('Prediction:', prediction[0])

    return render_template('result.html', prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
