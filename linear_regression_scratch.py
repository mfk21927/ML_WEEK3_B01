import numpy as np 
import matplotlib.pyplot as plt

# 1. creating data through random 
np.random.seed(42)
X = 2 * np.random.rand(100)
y = 2 * X + 1 + np.random.randn(100) * 0.5

# 2. Linear regression class
class LinearRegressionScratch: 
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def compute_cost(self, y, y_pred):
        # cost function (MSE)
        return np.mean((y - y_pred)**2)

    def fit(self, X, y):
        n = len(y)
        
        self.weights = 0
        self.bias = 0
        
        # Gradient descent logic inside fit()
        for _ in range(self.n_iterations):
            y_pred = self.predict(X)
            
            # Calculate gradients
            dw = (2/n) * np.sum(X * (y_pred - y))
            db = (2/n) * np.sum(y_pred - y)
            
            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
           
            cost = self.compute_cost(y, y_pred)
            self.cost_history.append(cost)
        
    def predict(self, X):
        
        return self.weights * X + self.bias



model = LinearRegressionScratch(learning_rate=0.1, n_iterations=100)
model.fit(X, y)
predictions = model.predict(X)

# 7. Calculate R-squared manually
ss_res = np.sum((y - predictions)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1 - (ss_res / ss_tot)
print(f"Manual R2 Score: {r2:.4f}")


plt.figure(figsize=(12, 5))

# Plot Data and Regression Line
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, predictions, color='red', label='Regression Line')
plt.title("Regression Fit")
plt.legend()

# Plot Cost Convergence
plt.subplot(1, 2, 2)
plt.plot(model.cost_history)
plt.title("Cost Function Convergence")
plt.xlabel("Iterations")
plt.ylabel("MSE")

plt.tight_layout()
plt.show()


