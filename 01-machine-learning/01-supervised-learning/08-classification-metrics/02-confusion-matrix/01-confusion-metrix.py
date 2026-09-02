import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


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


# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)


# Extract TN, FP, FN, TP
TN, FP, FN, TP = cm.ravel()

print("\nTrue Negative:", TN)
print("False Positive:", FP)
print("False Negative:", FN)
print("True Positive:", TP)


# Plot confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fail", "Pass"]
)

display.plot()

plt.title("Confusion Matrix")
plt.show()