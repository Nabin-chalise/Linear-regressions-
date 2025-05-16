# 🎯 Understanding Bias-Variance, Regularization & Ridge Regression

---

## 🔍 (a) Bias-Variance Explanation

When a **high-degree polynomial regression model** performs **exceptionally well on training data** but **poorly on test data**, it’s a classic case of **overfitting**.

> **Bias:**  
> The model has **low bias** because it fits the training data very closely (high flexibility).

> **Variance:**  
> The model has **high variance**, meaning it is too sensitive to noise or small fluctuations in training data and fails to generalize.

### ⚠️ Conclusion:  
Your model is **overfitting** due to **low bias and high variance**.

---

## 🛠️ (b) Regularization Techniques in Scikit-Learn

Regularization helps reduce overfitting by **penalizing model complexity** and limiting flexibility.

| Technique   | Scikit-Learn Class  | Penalty Term     | Effect                                  |
|-------------|---------------------|------------------|-----------------------------------------|
| **Ridge**   | `Ridge()`           | L2: λ ∑ w²       | Shrinks coefficients towards zero       |
| **Lasso**   | `Lasso()`           | L1: λ ∑ |w|      | Can zero out some coefficients (feature selection) |
| **ElasticNet** | `ElasticNet()`    | L1 + L2          | Combines Ridge and Lasso penalties       |

> These techniques improve **generalization** by controlling model complexity.

---

## 🚀 (c) Implementation Using Ridge Regression

Below is a Python example demonstrating:

- Overfitting of a high-degree polynomial model  
- How Ridge regularization improves generalization  

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic dataset
np.random.seed(42)
X = np.sort(np.random.rand(100, 1) * 2 - 1, axis=0)  # range [-1, 1]
y = np.sin(2 * np.pi * X).ravel() + 0.3 * np.random.randn(100)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# High-degree Polynomial Regression (Overfitting)
degree = 15
model_poly = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model_poly.fit(X_train, y_train)
y_pred_poly = model_poly.predict(X_test)

# Ridge Regularized Polynomial Regression
model_ridge = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1.0))
model_ridge.fit(X_train, y_train)
y_pred_ridge = model_ridge.predict(X_test)

# Evaluate performance
mse_poly = mean_squared_error(y_test, y_pred_poly)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

print(f"Test MSE - Plain Polynomial Regression: {mse_poly:.4f}")
print(f"Test MSE - Ridge Regression: {mse_ridge:.4f}")

# Plot results
plt.figure(figsize=(10, 5))
plt.scatter(X_test, y_test, color='black', label="Test Data")
x_plot = np.linspace(-1, 1, 200).reshape(-1, 1)
plt.plot(x_plot, model_poly.predict(x_plot), label="Plain Polynomial", color='red', linestyle='--')
plt.plot(x_plot, model_ridge.predict(x_plot), label="Ridge Regularized", color='blue')
plt.legend()
plt.title("High-degree Polynomial vs Ridge Regularization")
plt.xlabel("X")
plt.ylabel("y")
plt.grid(True)
plt.show()
