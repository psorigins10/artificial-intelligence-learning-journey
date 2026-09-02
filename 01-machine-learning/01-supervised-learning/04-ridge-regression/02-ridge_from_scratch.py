import numpy as np

# Training data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 8, 10])

# Ridge regularization strength
alpha = 1.0

# Means
x_mean = np.mean(X)
y_mean = np.mean(y)

# Calculate weight (w)
w = np.sum((X - x_mean) * (y - y_mean)) / (
    np.sum((X - x_mean) ** 2) + alpha
)

# Calculate intercept (b)
b = y_mean - w * x_mean

print("Weight:", w)
print("Intercept:", b)

# Prediction
y_pred = w * X + b

print("Predictions:", y_pred)