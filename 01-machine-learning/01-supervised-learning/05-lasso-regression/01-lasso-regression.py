import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, mean_squared_error

df = pd.read_csv("../../04-data-science/data-bases/house_price_lasso_practice.csv")

X = df.drop(["price_lakh"], axis = 1)

y = df["price_lakh"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.3,
    random_state = 42
)

reg = linear_model.Lasso(alpha = 1)
reg.fit(X_train, y_train)

print(f"Cofficient: {reg.coef_}")
print(f"\nIntercept: {reg.intercept_}")

y_pred = reg.predict(X_test)
print(f"\nPrediction: {y_pred}")

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print(f"\nR2: {r2}")
print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

import numpy as np

area_values = np.linspace(
    df["area_sqft"].min(),
    df["area_sqft"].max(),
    100
)

# Create 100 rows where every feature is at its mean
X_plot = pd.DataFrame(
    np.tile(X_train.mean().values, (100, 1)),
    columns=X_train.columns
)

# Change only area_sqft
X_plot["area_sqft"] = area_values

# Predict using the Ridge model
y_plot = reg.predict(X_plot)

# Plot actual data
plt.scatter(
    df["area_sqft"],
    df["price_lakh"],
    label="Actual"
)

# Plot Ridge predictions
plt.plot(
    area_values,
    y_plot,
    label="Lasso prediction"
)

plt.xlabel("Area (sqft)")
plt.ylabel("Price (₹ lakh)")
plt.title("Area vs Price — Lasso Regression")
plt.legend()

plt.show()