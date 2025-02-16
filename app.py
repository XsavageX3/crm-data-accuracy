from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("models/random_forest.pkl")

@app.route("/")
def home():
    return "CRM Data Cleaning API Running"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['file']
    data = pd.read_csv(file)

    predictions = model.predict(data.drop(columns=['CustomerID', 'Name', 'Email', 'Phone']))
    data['Predictions'] = predictions

    return data.to_json(orient="records")

if __name__ == "__main__":
    app.run(debug=True)
