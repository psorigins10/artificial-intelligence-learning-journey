from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt

X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_informative=1,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=1,
    random_state=41,
    hypercube=False,
    class_sep=10
)

# Train perceptron
model = Perceptron()
model.fit(X, y)

# Parameters learned by the model
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Get w0, w1, w2
intercept = model.intercept_[0]
coef = model.coef_[0]

# w0 + w1*x1 + w2*x2 = 0
#
# x2 = -(w1/w2)*x1 - (w0/w2)

m = -(coef[0] / coef[1])
b = -(intercept / coef[1])

x_input = np.linspace(-3, 3, 100)
y_input = m * x_input + b

# Plot
plt.figure(figsize=(10, 6))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap='winter',
    s=100
)

plt.plot(
    x_input,
    y_input,
    color='red',
    linewidth=3
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Perceptron Decision Boundary")
plt.ylim(-3, 2)

plt.show()