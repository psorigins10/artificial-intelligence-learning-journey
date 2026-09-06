import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error

np.random.seed(42)

X = np.linspace(-10, 10, 120).reshape(-1, 1)

y = (
    4 * X.flatten()**2
    - 7 * X.flatten()
    + 20
    + np.random.normal(0, 25, 120)
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)

model = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("linear", LinearRegression())
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

X_curve = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_curve = model.predict(X_curve)

df = pd.DataFrame({
    "Test Values" : X_test.flatten(),
    "Actual Values" : y_test,
    "Predicted Values" : y_pred
})

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print(df)

print(f"\nR2 Score: {r2}")
print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

plt.scatter(X, y)
plt.plot(X_curve, y_curve)

plt.show()