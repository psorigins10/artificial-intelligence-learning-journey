import numpy as np
from sklearn.metrics import log_loss


# Actual values
y_true = np.array([
    1, 1, 0, 0, 1
])

# Predicted probabilities for class 1
y_probability = np.array([
    0.9, 0.8, 0.2, 0.1, 0.7
])


# Calculate Log Loss
loss = log_loss(y_true, y_probability)

print("Log Loss:", loss)