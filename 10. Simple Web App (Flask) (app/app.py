from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("../models/model.pkl", "rb"))

@app.route("/")
def home():
    return "House Price Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    prediction = model.predict([data])
    return jsonify({"price": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
