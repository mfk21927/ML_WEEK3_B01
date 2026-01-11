import joblib
import pickle
import json
import time
import numpy as np

# Sample data for prediction (first row of California dataset)
sample_input = np.array([[8.32, 41.0, 6.98, 1.02, 322.0, 2.55, 37.88, -122.23]])

def measure_load(name, load_func):
    start = time.time()
    model = load_func()
    end = time.time()
    prediction = model.predict(sample_input)
    print(f"{name:<10} | Time: {end-start:.6f}s | Prediction: {prediction[0]:.4f}")

#Load Pickle
measure_load("Pickle", lambda: pickle.load(open('model.pkl', 'rb')))

#Load Joblib
measure_load("Joblib", lambda: joblib.load('model.joblib'))

#Load and Reconstruct from JSON
start = time.time()
with open('model.json', 'r') as f:
    weights = json.load(f)
# Manual prediction: y = X * coef + intercept
json_prediction = np.dot(sample_input, np.array(weights['coef'])) + weights['intercept']
end = time.time()
print(f"{'JSON':<10} | Time: {end-start:.6f}s | Prediction: {json_prediction[0]:.4f}")