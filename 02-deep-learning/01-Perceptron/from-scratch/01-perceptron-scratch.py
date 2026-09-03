from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt


def step(z):
    return 1 if z>0 else 0


def perceptron(X, y):
    X = np.insert(X, 0, 1, axis=1)

    weights = np.ones(X.shape[1])
    learningRate = 0.1

    for epoch in range(1000):
        errors = 0

        for j in range(X.shape[0]):
            y_hat = step(np.dot(X[j], weights))

            if y[j] != y_hat:
                weights += learningRate * (y[j] - y_hat) * X[j]
                errors += 1

        if errors == 0:
            print("Converged at epoch:", epoch)
            break

    return weights[0], weights[1:]

X, y = make_classification(
    n_samples = 200,
    n_features = 2,
    n_informative = 1,
    n_redundant = 0,
    n_classes = 2,
    n_clusters_per_class = 1,
    random_state = 41,
    hypercube = False,
    class_sep = 10
)

intercept_, coef_ = perceptron(X, y)
print("Intercept: ", intercept_)
print("Coef: ", coef_)

m = -(coef_[0] / coef_[1])
b = -(intercept_ / coef_[1])

x_input = np.linspace(-3, 3, 100)
y_input = m * x_input + b

plt.figure(figsize = (10, 6))
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = 'winter', s = 100)
plt.plot(x_input, y_input, color = 'red', linewidth = 3)
plt.ylim(-3, 2)
plt.show()