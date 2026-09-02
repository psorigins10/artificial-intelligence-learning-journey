from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from source import LinearRegression

X, y = make_regression(
    n_samples = 100,
    n_features = 10,
    n_informative = 3,
    random_state = 42,
    noise = 10
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)

model = LinearRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Predictions: ", y_pred)
print("\nR2-Score: ", r2_score(y_test, y_pred))
print("Coef: ", model.coef_)
print("Intercept: ", model.intercept_)