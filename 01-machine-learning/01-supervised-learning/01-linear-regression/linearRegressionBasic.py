from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

# Dataset
data = {
    "landDimentions": [100, 200, 300, 400, 500, 600, 700, 800],
    "landPrice": [650000, 980000, 1350000, 1500000,
                  2100000, 2150000, 2700000, 2950000]
}

# Creating a dataframe using dataset
df = pd.DataFrame(data)

# Assining X, y parameters
X = df[["landDimentions"]]
y = df["landPrice"]

model = LinearRegression() # Creating a LinearRegression model instance
model.fit(X, y) # Passign the parameters so model can actually train

value : int = int(input("Enter Land Dimentions to predict price(meters sqrt): "))

prediction = model.predict([[value]]) # Asking the trained model to predict the price

print(f"The price for {value} is {prediction}")

# Data Visualization
plt.plot(X, y)
plt.grid(alpha= 0.5)
plt.show()