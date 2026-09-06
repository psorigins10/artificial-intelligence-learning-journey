import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error

X, y = make_regression(
    n_features = 2,
    random_state = 0,
    n_samples = 100
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)

reg = ElasticNetCV(random_state = 0)
reg.fit(X_train, y_train)

print(reg.coef_)
print(reg.intercept_)

y_pred = reg.predict(X_test)
print(f"\n{y_pred}")

print("\nR²:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", root_mean_squared_error(y_test, y_pred))

plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Elastic Net: Actual vs Predicted")

plt.show()