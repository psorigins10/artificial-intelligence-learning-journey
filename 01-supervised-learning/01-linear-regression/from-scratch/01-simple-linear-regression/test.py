from source import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


data = {
    "area": [
        450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
        950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400,
        1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900,
        1950, 2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400,
        2450, 2500, 2550, 2600, 2650, 2700, 2800, 2900, 3000, 3200
    ],

    "price": [
        23, 25, 29, 31, 34, 36, 40, 38, 43, 46,
        48, 45, 51, 54, 53, 55, 59, 62, 64, 66,
        69, 68, 73, 76, 78, 81, 79, 82, 86, 89,
        91, 90, 95, 98, 101, 103, 105, 108, 110, 114,
        116, 115, 121, 123, 126, 129, 133, 137, 142, 151
    ]
}


df = pd.DataFrame(data)

X = df.iloc[:, 0].values
y = df.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred, coff, intercept = model.predict(X_test)
print("Predicted Values: ", y_pred)
print("Actual Values: ", y_test)

print("Cofficeient: ", coff)
print("Intercept: ", intercept)

mae = np.mean(np.abs(y_test - y_pred))
mse = np.mean((y_test - y_pred) ** 2)

print("MAE:", mae)
print("MSE:", mse)

