import pickle
import numpy as np

def load_model(path="models/model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def predict(input_data):
    model = load_model()
    input_array = np.array(input_data).reshape(1, -1)
    return model.predict(input_array)
