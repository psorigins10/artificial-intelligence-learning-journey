from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from source import LinearRegressionGD
import numpy as np

X, y = make_regression(
    n_samples = 200,
    n_features = 1,
    n_informative = 1,
    random_state = 11,
    noise = 40
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)

real_coef_ = 62.60428295

model = LinearRegressionGD()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(y_pred)

print("\nCoef: ", model.coef_)
print("Intercept: ", model.intercept_)
print("OLS Coef: ", real_coef_)

x_line = np.linspace(X_train.min(), X_train.max(), 100)
y_line = model.predict(x_line.reshape(-1, 1))

plt.figure(figsize=(9, 6))

plt.scatter(
    X_train,
    y_train,
    marker=".",
    s=70,
    alpha=0.7,
    label="Training data"
)

plt.plot(
    x_line,
    y_line,
    linewidth=2.5,
    label="Gradient Descent Fit"
)

plt.title("Linear Regression using Gradient Descent", fontsize=15)
plt.xlabel("X")
plt.ylabel("y")

plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

plt.show()