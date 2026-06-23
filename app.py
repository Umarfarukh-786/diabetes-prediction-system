from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("models/diabetes_model.pkl")
scaler = joblib.load("models/scaler.pkl")
imputer = joblib.load("models/imputer.pkl")

print("Model Loaded Successfully")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    pregnancies = float(request.form['pregnancies'])
    glucose = float(request.form['glucose'])
    bp = float(request.form['bp'])
    skin = float(request.form['skin'])
    insulin = float(request.form['insulin'])
    bmi = float(request.form['bmi'])
    dpf = float(request.form['dpf'])
    age = float(request.form['age'])

    data = np.array([[pregnancies, glucose, bp,
                      skin, insulin, bmi,
                      dpf, age]])

    print("Input Data =", data)

    data = imputer.transform(data)
    data = scaler.transform(data)

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0]
    confidence = round(max(probability) * 100, 2)

    if prediction == 1:
        result = "Diabetic"
    else:
        result = "Non-Diabetic"

    print("Prediction =", prediction)
    print("Result =", result)
    print("Confidence =", confidence)

    return render_template(
        'result.html',
        prediction=result,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)