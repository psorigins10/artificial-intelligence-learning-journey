import numpy as np

# Actual values
y = np.array([10, 20, 30, 40, 50])

# Predicted values
y_pred = np.array([12, 18, 33, 37, 55])

# Calculate errors
errors = y - y_pred

# Square the errors
squared_errors = errors ** 2

# Calculate MSE
mse = np.mean(squared_errors)

# Calculate RMSE
rmse = np.sqrt(mse)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")