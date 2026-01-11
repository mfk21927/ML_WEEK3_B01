import joblib
import pickle
import json
import os
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

#  Train a model on California Housing
data = fetch_california_housing()
X, y = data.data, data.target
model = LinearRegression()
model.fit(X, y)

#  Save with Pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

#  Saving with Joblib
joblib.dump(model, 'model.joblib')

#  Saving weights as JSON 
model_weights = {
    "coef": model.coef_.tolist(),
    "intercept": model.intercept_
}
with open('model.json', 'w') as f:
    json.dump(model_weights, f)

#  Document file sizes
print(f"Pickle Size: {os.path.getsize('model.pkl')} bytes")
print(f"Joblib Size: {os.path.getsize('model.joblib')} bytes")
print(f"JSON Size:   {os.path.getsize('model.json')} bytes")