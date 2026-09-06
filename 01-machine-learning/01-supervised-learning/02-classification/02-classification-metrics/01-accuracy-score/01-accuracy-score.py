import numpy as np
from sklearn.metrics import accuracy_score

y_actual = np.array([0, 1, 1, 1, 0, 0, 1, 0])
y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])

print(f"Accuracy Score: {accuracy_score(y_actual, y_pred)}")

y_actual = np.array([0, 1, 1, 1, 0, 1, 1, 0])
y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])

print(f"Accuracy Score: {accuracy_score(y_actual, y_pred)}")


#CHECK README FOR THEORY