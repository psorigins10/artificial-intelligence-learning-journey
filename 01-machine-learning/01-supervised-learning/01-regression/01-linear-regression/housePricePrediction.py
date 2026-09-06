import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    "area": [500, 800, 1000, 1200, 1500, 1800, 2000, 2500],
    "bedrooms": [1, 2, 2, 3, 3, 4, 4, 5],
    "price": [25, 38, 45, 55, 68, 82, 90, 115]
}

df = pd.DataFrame(data)

X = df[["area", "bedrooms"]]
y = df["price"]

model = LinearRegression()
model.fit(X,y)

area = int(input("Enter Area for the house: "))
bedrooms = int(input("Enter number of bedrooms in house: "))

new_house = pd.DataFrame({
    "area" : [area],
    "bedrooms" : [bedrooms]
})
prediction = model.predict(new_house)[0]

print(f"Predicted price: ₹{prediction:.2f} lakh")
r2 = model.score(X, y)
print(f"R²: {r2:.4f}")

predictions = model.predict(X)


fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Area vs Price
ax[0].scatter(df["area"], df["price"])
ax[0].plot(df["area"], predictions)
ax[0].set_xlabel("Area")
ax[0].set_ylabel("Price")
ax[0].set_title("Area vs Price")

# Bedrooms vs Price
ax[1].scatter(df["bedrooms"], df["price"])
ax[1].set_xlabel("Bedrooms")
ax[1].set_ylabel("Price")
ax[1].set_title("Bedrooms vs Price")

plt.tight_layout()
plt.show()