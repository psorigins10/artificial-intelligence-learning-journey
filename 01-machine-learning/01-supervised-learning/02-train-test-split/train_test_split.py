import numpy as np # type: ignore[reportMissingImports]
import pandas as pd  # type: ignore[reportMissingImports]
import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
from sklearn.linear_model import LinearRegression  # type: ignore[reportMissingImports]
from sklearn.model_selection import train_test_split  # type: ignore[reportMissingImports]
from sklearn.metrics import r2_score  # type: ignore[reportMissingImports]


######################################
########This is bit noisy data########
######################################

distance_in_Km = np.array([
    3, 7, 12, 18, 25, 31, 38, 44, 51, 57,
    64, 70, 77, 83, 91, 98, 105, 113, 120, 128,
    135, 143, 151, 159, 168, 176, 185, 193, 202, 211,
    220, 229, 238, 247, 256, 265, 274, 283, 292, 301,
    310, 319, 328, 337, 346, 355, 364, 373, 382, 391,
    400, 409, 418, 427, 436, 445, 454, 463, 472, 481,
    490, 499, 508, 517, 526, 535, 544, 553, 562, 571,
    580, 589, 598, 607, 616, 625, 634, 643, 652, 661,
    670, 679, 688, 697, 706, 715, 724, 733, 742, 751,
    760, 769, 778, 787, 796, 805, 814, 823, 832, 841,
    850, 859, 868, 877, 886, 895, 904, 913, 922, 931
])

price = np.array([
    21, 34, 37, 61, 68, 91, 88, 119, 113, 147,
    142, 176, 169, 203, 194, 231, 224, 264, 251, 291,
    278, 321, 307, 351, 337, 382, 365, 411, 398, 442,
    425, 476, 459, 501, 488, 536, 519, 561, 548, 601,
    579, 631, 615, 663, 647, 701, 682, 729, 714, 767,
    748, 802, 786, 838, 821, 879, 857, 914, 895, 951,
    934, 987, 973, 1031, 1008, 1069, 1051, 1111, 1088, 1147,
    1132, 1191, 1170, 1234, 1210, 1271, 1255, 1318, 1297, 1359,
    1341, 1408, 1382, 1449, 1431, 1498, 1474, 1543, 1521, 1588,
    1567, 1632, 1610, 1681, 1659, 1727, 1704, 1776, 1751, 1823
])

df = pd.DataFrame({
    "Distance" : distance_in_Km[:100],
    "Price" : price
})

X = df[["Distance"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)

model = LinearRegression()
model.fit(X_train, y_train)

print("Cofficent: ", model.coef_[0])
print("Intercept: ", model.intercept_)

y_pred = model.predict(X_test)
print("Prediction: ", y_pred)

r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2}")

plt.scatter(X, y)

# Regression line
X_sorted = X.sort_values(by="Distance")
y_line = model.predict(X_sorted)

plt.plot(X_sorted, y_line)

# Intercept point
plt.scatter(
    0,
    model.intercept_,
    label=f"Intercept = {model.intercept_:.2f}"
)

plt.xlabel("Distance")
plt.ylabel("Price")
plt.legend()
plt.show()