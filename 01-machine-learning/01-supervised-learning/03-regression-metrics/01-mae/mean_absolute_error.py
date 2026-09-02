import numpy as np

# Actual values
y = np.array([10, 20, 30, 40, 50])

# Predicted values
y_pred = np.array([12, 18, 33, 37, 55])

# Calculate errors
errors = y - y_pred

# Get absolute errors
absolute_errors = np.abs(errors)

# Calculate MAE
mae = np.mean(absolute_errors)

print(f"Errors: {errors}")
print(f"Absolute Errors: {absolute_errors}")
print(f"MAE: {mae}")


# We can also use sklearn.metrics mean_squared_error