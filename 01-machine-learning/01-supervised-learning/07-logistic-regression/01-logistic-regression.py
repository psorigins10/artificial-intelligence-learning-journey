import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X = np.array([
    0.5, 0.7, 0.9, 1.0, 1.2,
    1.4, 1.5, 1.7, 1.8, 2.0,
    2.1, 2.2, 2.4, 2.5, 2.6,
    2.8, 2.9, 3.0, 3.1, 3.2,
    3.3, 3.4, 3.5, 3.6, 3.7,
    3.8, 3.9, 4.0, 4.1, 4.2,
    4.3, 4.5, 4.6, 4.8, 5.0,
    5.2, 5.4, 5.5, 5.7, 6.0,
    6.2, 6.5, 6.8, 7.0, 7.2,
    7.5, 8.0, 8.2, 8.5, 9.0
])

y = np.array([
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1
])

X = X.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(y_pred)

X_curve = np.linspace(0, 9, 200).reshape(-1, 1)

y_probability = clf.predict_proba(X_curve)[:, 1]


accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)


cm = confusion_matrix(y_test, y_pred)
print("Confusion Metrics",cm)


# Plot actual data
plt.scatter(X, y, label="Actual data")

# Plot logistic regression curve
plt.plot(X_curve, y_probability, label="Logistic curve")

# Decision boundary
plt.axhline(0.5, linestyle="--", label="0.5 threshold")

plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression: Study Hours vs Pass")

plt.yticks([0, 0.5, 1], ["Fail", "0.5", "Pass"])

plt.legend()
plt.grid()

plt.show()