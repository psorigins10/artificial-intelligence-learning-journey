import numpy as np

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# Actual values
y_test = np.array([
    0, 0, 0, 1, 1,
    1, 1, 0, 1, 0
])

# Model predictions
y_pred = np.array([
    0, 0, 1, 1, 1,
    0, 1, 0, 1, 0
])


# Calculate Precision
precision = precision_score(y_test, y_pred)

# Calculate Recall
recall = recall_score(y_test, y_pred)

# Calculate F1 Score
f1 = f1_score(y_test, y_pred)


print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)