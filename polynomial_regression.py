import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#  Creating synthetic non-linear dataset
np.random.seed(42)
X = 2 - 3 * np.random.normal(0, 1, 100).reshape(-1, 1)
y = X - 2 * (X ** 2) + 0.5 * (X ** 3) + np.random.normal(-3, 3, 100).reshape(-1, 1)

# Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#  Degrees 
degrees = [1, 2, 3, 5, 10]

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='black', s=15, label="Actual Data")

# Prepare  line for plotting the curves
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

print(f"{'Degree':<10} | {'Train RMSE':<15} | {'Test RMSE':<15}")
print("-" * 45)

for degree in degrees:
    #  Transform and Fit
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # 6 Calculating errors
    train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train_poly)))
    test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test_poly)))
    
    print(f"{degree:<10} | {train_rmse:<15.4f} | {test_rmse:<15.4f}")

    #  Plotting 
    y_range_pred = model.predict(poly.transform(X_range))
    plt.plot(X_range, y_range_pred, label=f"Degree {degree}")



plt.title("Polynomial Regression: Degree Comparison")
plt.xlabel("X Feature")
plt.ylabel("Target Y")
plt.legend()
plt.ylim(y.min()-5, y.max()+5)
plt.show()