# Elastic Net Regression

A small hands-on experiment with **Elastic Net Regression** using scikit-learn.

The goal of this experiment was to understand how Elastic Net combines **L1 (Lasso)** and **L2 (Ridge)** regularization, then evaluate the model on unseen test data.

---

## What is Elastic Net?

Elastic Net combines L1 and L2 regularization.

Conceptually:

```text
Elastic Net = Lasso + Ridge
```

The objective can be written as:

\[
\text{MSE} + \lambda\left(\alpha\|w\|_1 + (1-\alpha)\|w\|_2^2\right)
\]

In scikit-learn, the parameters are exposed as:

- `alpha` → overall regularization strength
- `l1_ratio` → controls the L1/L2 mixture
  - `l1_ratio = 1` → Lasso-like
  - `l1_ratio = 0` → Ridge-like
  - values between 0 and 1 → mixture of both

### Why use Elastic Net?

- **Lasso (L1)** can shrink coefficients to exactly zero, providing feature selection.
- **Ridge (L2)** shrinks coefficients and tends to behave more stably with correlated features.
- **Elastic Net** combines both behaviors.

---

## Dataset

For this experiment, I used scikit-learn's synthetic regression dataset generator:

```python
from sklearn.datasets import make_regression

X, y = make_regression(
    n_features=2,
    random_state=0,
    n_samples=100
)
```

This creates:

- 100 samples
- 2 features
- a continuous regression target

`make_regression()` is useful for learning because it lets me focus on the ML algorithm without spending time cleaning a dataset.

---

## Train/Test Split

I split the data into training and testing sets:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

That gives:

```text
80 samples → training
20 samples → testing
```

The model only sees `X_train` and `y_train` during training.

The test set is kept unseen until prediction/evaluation.

---

## Model

```python
from sklearn.linear_model import ElasticNet

reg = ElasticNet(random_state=0)

reg.fit(X_train, y_train)
```

After training, the learned parameters were:

```text
Coefficients:
[19.68300791 62.84236797]

Intercept:
1.9451555090470225
```

So approximately:

\[
\hat{y} =
19.683x_1 +
62.842x_2 +
1.945
\]

---

## Predictions

Predictions were generated using the unseen test features:

```python
y_pred = reg.predict(X_test)
```

The model produced:

```text
[-39.80498456, -91.31648712, -62.11542397, 66.08762348,
 -145.62558993, -33.52038601, 72.48099311, -14.58048231,
 -54.04918053, 127.19003121, -53.39344153, 66.6553401,
 -46.71267871, 115.95561971, 8.62661167, -14.15956595,
 17.83297176, 3.54045201, -49.63328343, 137.14066709]
```

---

## Evaluation

I evaluated the model using R², MAE, MSE, and RMSE.

| Metric | Result |
|---|---:|
| R² | 0.8795 |
| MAE | 32.8079 |
| MSE | 1547.1495 |
| RMSE | 39.3338 |

### R²

```text
R² = 0.8795
```

The model explains approximately **87.95% of the variance** in the test target relative to the mean-prediction baseline.

This is **not** the same thing as saying the model is 87.95% accurate.

### MAE

```text
MAE = 32.81
```

On average, the absolute prediction error was about 32.8 target units.

### MSE

```text
MSE = 1547.15
```

MSE squares the errors, so its numerical value is in squared target units.

### RMSE

```text
RMSE = 39.33
```

RMSE is easier to interpret because it returns to the same units as the target.

Also:

\[
\sqrt{1547.15} \approx 39.33
\]

---

## Actual vs Predicted

I plotted the actual test values against the model's predictions.

The diagonal line represents:

\[
y_{\text{pred}} = y_{\text{actual}}
\]

Points closer to the diagonal represent smaller prediction errors.

The experiment also showed that the model's extreme predictions were pulled toward zero, which is consistent with the coefficient-shrinking behavior introduced by regularization.

---

## What I learned

- Elastic Net combines **L1 and L2 regularization**.
- L1 encourages sparse coefficients and can perform feature selection.
- L2 shrinks coefficients without generally forcing them to zero.
- `alpha` controls the overall regularization strength in scikit-learn.
- `l1_ratio` controls the L1/L2 mixture.
- `fit()` learns the model parameters.
- `predict()` generates predictions for new data.
- A train/test split lets me evaluate generalization on unseen samples.
- R², MAE, MSE, and RMSE measure different aspects of regression performance.
- MSE can look numerically large because errors are squared.
- Actual-vs-predicted plots help visualize model error.

---

## Code

```python
import matplotlib.pyplot as plt

from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error
)

X, y = make_regression(
    n_features=2,
    random_state=0,
    n_samples=100
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

reg = ElasticNet(random_state=0)

reg.fit(X_train, y_train)

print(reg.coef_)
print(reg.intercept_)

y_pred = reg.predict(X_test)

print("R²:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", root_mean_squared_error(y_test, y_pred))

plt.scatter(y_test, y_pred)

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val]
)

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Elastic Net: Actual vs Predicted")

plt.show()
```
