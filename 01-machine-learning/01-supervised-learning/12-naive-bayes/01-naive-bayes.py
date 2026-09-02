import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt


df = pd.read_csv("../04-data-science/data-bases/pokemon.csv")


X = df[
    [
        "HP",
        "Attack",
        "Defense",
        "Speed",
    ]
]

y = df["Legendary"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


gnb = GaussianNB()

y_pred = gnb.fit(X_train, y_train).predict(X_test)


print(f"Predictions: {y_pred}")

print(f"\n{classification_report(y_test, y_pred)}")

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")


cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)


TN, FP, FN, TP = cm.ravel()

print("\nTrue Negative:", TN)
print("False Positive:", FP)
print("False Negative:", FN)
print("True Positive:", TP)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["NON-L", "Legendary"]
)

display.plot()

plt.title("Confusion Matrix")
plt.show()