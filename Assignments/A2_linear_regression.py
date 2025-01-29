import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Creating DataFrame using numpy arrays
ages = np.array([22, 25, 30, 35, 40, 45, 50, 55, 60, 65]).reshape(-1, 1)
salaries = np.array([25000, 28000, 35000, 40000, 50000, 60000, 70000, 80000, 85000, 90000])

df = pd.DataFrame({"Age": ages.flatten(), "Salary": salaries})

# Splitting dataset (Using train_size instead of test_size)
X_train, X_test, y_train, y_test = train_test_split(ages, salaries, train_size=0.8, random_state=42)

# Using np.linalg.lstsq instead of sklearn for fitting the model
A = np.hstack([X_train, np.ones_like(X_train)])  # Adding bias term
coeff, _, _, _ = np.linalg.lstsq(A, y_train, rcond=None)

# Predicting values
y_pred = coeff[0] * ages.flatten() + coeff[1]

# Plotting the results
plt.scatter(ages, salaries, color='blue', label="Actual Data")
plt.plot(ages, y_pred, color='red', label="Regression Line")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Simple Linear Regression: Age vs Salary")
plt.legend()
plt.show()
