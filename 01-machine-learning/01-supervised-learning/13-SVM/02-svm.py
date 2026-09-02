import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report


# Create 2D classification data
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    class_sep=1.5,
    random_state=42
)


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Linear SVM
clf = make_pipeline(
    StandardScaler(),
    SVC(kernel="linear")
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(y_pred)

print(f"\n{classification_report(y_test, y_pred)}")

# Get the actual SVM model from the pipeline
svc = clf.named_steps["svc"]
scaler = clf.named_steps["standardscaler"]


# Scale the training data for plotting
X_train_scaled = scaler.transform(X_train)


# Get support vectors
support_vectors = svc.support_vectors_


# SVM equation:
# w1*x1 + w2*x2 + b = 0
w = svc.coef_[0]
b = svc.intercept_[0]


# Create x values
x = np.linspace(
    X_train_scaled[:, 0].min() - 1,
    X_train_scaled[:, 0].max() + 1,
    100
)


# Decision boundary
y_boundary = -(w[0] * x + b) / w[1]


# Margin boundaries
margin = 1 / np.linalg.norm(w)

y_margin_upper = y_boundary + margin
y_margin_lower = y_boundary - margin


# Plot classes
plt.figure(figsize=(10, 7))

plt.scatter(
    X_train_scaled[y_train == 0, 0],
    X_train_scaled[y_train == 0, 1],
    label="Class 0"
)

plt.scatter(
    X_train_scaled[y_train == 1, 0],
    X_train_scaled[y_train == 1, 1],
    label="Class 1"
)


# Plot decision boundary
plt.plot(
    x,
    y_boundary,
    label="Decision Boundary"
)


# Plot margins
plt.plot(
    x,
    y_margin_upper,
    "--",
    label="Margin"
)

plt.plot(
    x,
    y_margin_lower,
    "--"
)


# Plot support vectors
plt.scatter(
    support_vectors[:, 0],
    support_vectors[:, 1],
    s=150,
    facecolors="none",
    edgecolors="black",
    linewidths=2,
    label="Support Vectors"
)


plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Linear SVM: Hyperplane, Margins, and Support Vectors")
plt.legend()
plt.grid(True)

plt.show()