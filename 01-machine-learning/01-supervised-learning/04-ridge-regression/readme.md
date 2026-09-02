# Ridge Regression

Ridge Regression is a type of **linear regression** that uses **L2 regularization** to reduce the impact of large model coefficients.

It is especially useful when a dataset has many features, correlated features, or when ordinary Linear Regression is prone to overfitting.

---

## 1. What is Ridge Regression?

Ridge Regression uses the same linear prediction equation as Linear Regression:

$$
\hat{y} = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b
$$

The difference is how the model chooses the weights.

### Linear Regression

Linear Regression tries to minimize the prediction error:

$$
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

### Ridge Regression

Ridge adds a penalty for large weights:

$$
\boxed{
MSE + \alpha\sum_{j=1}^{n}w_j^2
}
$$

The second part is the **L2 regularization penalty**.

---

## 2. What is L2 Regularization?

L2 regularization adds the squared values of the coefficients to the loss function:

$$
\sum w_j^2
$$

So the Ridge objective becomes:

$$
\boxed{
MSE + \alpha(w_1^2+w_2^2+\cdots+w_n^2)
}
$$

This makes large coefficients more expensive.

Ridge therefore tries to find a balance between:

1. Making accurate predictions
2. Keeping the coefficients reasonably small

---

## 3. What is Alpha?

`alpha` controls the strength of the regularization.

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
```

### Small Alpha

```python
alpha = 0.01
alpha = 0.1
```

A small alpha means **weak regularization**.

The model is allowed to have relatively larger coefficients and behaves more like ordinary Linear Regression.

### Large Alpha

```python
alpha = 10
alpha = 100
```

A large alpha means **strong regularization**.

The model receives a stronger penalty for having large coefficients, so the coefficients are pushed closer toward zero.

### What happens as Alpha increases?

```text
Small Alpha
    ↓
Weak regularization
    ↓
Less coefficient shrinkage
    ↓
More flexible model

Large Alpha
    ↓
Strong regularization
    ↓
More coefficient shrinkage
    ↓
Less flexible model
```

If `alpha` becomes too large, the model can become too constrained and **underfit** the data.

If `alpha` is too small, the regularization effect becomes weak and the model behaves more like ordinary Linear Regression.

The goal is to find an `alpha` that gives good performance on **unseen data**.

---

## 4. Why Use Ridge Regression?

### Many Features

When a model has many features, some coefficients can become large or unstable.

Ridge helps control them.

### Correlated Features

If several features contain similar information, ordinary Linear Regression can produce unstable coefficients.

Ridge helps control this problem by shrinking the coefficients.

### Overfitting

If a Linear Regression model fits the training data very well but performs worse on unseen data, Ridge regularization may improve generalization.

---

## 5. Ridge vs Linear Regression

| Feature | Linear Regression | Ridge Regression |
|---|---|---|
| Linear model | Yes | Yes |
| Uses linear prediction equation | Yes | Yes |
| Regularization | No | L2 |
| Penalizes large weights | No | Yes |
| Uses alpha | No | Yes |
| Can reduce overfitting | Limited | Yes |
| Shrinks coefficients | No penalty | Yes |

Ridge is not a completely different prediction equation.

Both models make predictions using a linear combination of features.

The main difference is the **objective used to find the coefficients**.

---

## 6. Simple Ridge Example

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

The model predicts:

$$
\hat{y} = w_1x_1+w_2x_2+\cdots+w_nx_n+b
$$

while minimizing:

$$
MSE+\alpha\sum w_j^2
$$

---

## 7. Ridge Regression From Scratch

For a simple one-feature case:

$$
w =
\frac{
\sum (x_i-\bar{x})(y_i-\bar{y})
}{
\sum (x_i-\bar{x})^2+\alpha
}
$$

```python
import numpy as np

X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 8, 10])

alpha = 1.0

x_mean = np.mean(X)
y_mean = np.mean(y)

w = np.sum(
    (X - x_mean) * (y - y_mean)
) / (
    np.sum((X - x_mean) ** 2) + alpha
)

b = y_mean - w * x_mean

y_pred = w * X + b

print("Weight:", w)
print("Intercept:", b)
print("Predictions:", y_pred)
```

The important difference from ordinary Linear Regression is the addition of:

```python
+ alpha
```

in the denominator.

---

## 8. Dataset Used for Practice

The practice dataset contains:

- 20 observations
- 10 input features
- 1 target variable

### Features

| Feature | Description |
|---|---|
| `area_sqft` | House area in square feet |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `age_years` | Age of the house |
| `distance_city_km` | Distance from city center |
| `parking_spaces` | Number of parking spaces |
| `floor` | Floor number |
| `balcony_sqft` | Balcony area |
| `near_metro` | Whether the house is near a metro |
| `renovation_score` | Condition/renovation score |
| `price_lakh` | House price in ₹ lakh |

Dataset:

```text
house_price_ridge_practice.csv
```

---

## 9. Loading the Dataset

```python
import pandas as pd

df = pd.read_csv(
    "../../04-data-science/data-bases/house_price_ridge_practice.csv"
)

print(df)
```

Separate features and target:

```python
X = df.drop(["price_lakh"], axis=1)
y = df["price_lakh"]
```

---

## 10. Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)
```

The model learns from the training data and is evaluated using the test data.

---

## 11. Training the Ridge Model

```python
from sklearn import linear_model

reg = linear_model.Ridge(alpha=1)

reg.fit(X_train, y_train)

print("Coefficients:", reg.coef_)
print("Intercept:", reg.intercept_)
```

---

## 12. Making Predictions

```python
y_pred = reg.predict(X_test)

print(y_pred)
```

The prediction is based on:

$$
\hat{y} =
w_1x_1+w_2x_2+\cdots+w_{10}x_{10}+b
$$

---

## 13. Evaluating the Model

```python
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error
)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print("R²:", r2)
print("MSE:", mse)
print("MAE:", mae)
print("RMSE:", rmse)
```

### R² Score

Higher is generally better.

$$
R^2 =
1 -
\frac{\sum(y_i-\hat{y}_i)^2}
{\sum(y_i-\bar{y})^2}
$$

### MSE

Lower is better.

$$
MSE =
\frac{1}{n}
\sum(y_i-\hat{y}_i)^2
$$

### MAE

Lower is better.

$$
MAE =
\frac{1}{n}
\sum|y_i-\hat{y}_i|
$$

### RMSE

Lower is better.

$$
RMSE=\sqrt{MSE}
$$

---

## 14. Experimenting With Alpha

Example values:

```python
alphas = [0.01, 0.1, 1, 10, 100]
```

As alpha increases:

```text
alpha increases
      ↓
stronger L2 penalty
      ↓
smaller coefficients
      ↓
less flexible model
```

If alpha becomes too large, the model may underfit.

If alpha is too small, the regularization effect becomes weak and the model behaves more like Linear Regression.

The best alpha should be selected based on performance on unseen data.

---

## 15. Visualizing the Model

### Actual vs Predicted

```python
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price (₹ lakh)")
plt.ylabel("Predicted Price (₹ lakh)")
plt.title("Ridge Regression: Actual vs Predicted")

plt.show()
```

A good model should have predictions close to the diagonal relationship between actual and predicted values.

### Feature Visualization

```python
plt.scatter(df["area_sqft"], df["price_lakh"])

plt.xlabel("Area (sqft)")
plt.ylabel("Price (₹ lakh)")
plt.title("Area vs Price")

plt.show()
```

Individual feature plots help understand the dataset, but they do not represent the complete Ridge model because Ridge uses all 10 features simultaneously.

---

## 16. Important Note About the Dataset

The dataset contains only 20 observations and 10 features.

Therefore, the evaluation metrics should not be treated as evidence that this is a production-quality house-price model.

The dataset is primarily designed for learning:

- Ridge Regression
- L2 regularization
- Coefficients
- Alpha
- Train-test splitting
- Model evaluation
- Visualization

A real house-price model would require a much larger and more representative dataset.

---

## 17. Key Takeaways

- Ridge Regression is **Linear Regression with L2 regularization**.
- Ridge adds a penalty based on the squared coefficients.
- `alpha` controls the strength of regularization.
- Increasing `alpha` generally shrinks coefficients toward zero.
- Too much regularization can cause underfitting.
- Ridge is useful when features are correlated or when regularization is needed.
- Ridge does not automatically perform better than Linear Regression.
- Model performance should be evaluated on unseen data.
- R², MSE, MAE, and RMSE can be used to evaluate predictions.
- For small datasets, cross-validation is useful when selecting `alpha`.

---

## Core Idea

$$
\boxed{
\text{Ridge Regression}
=
\text{Linear Regression}
+
\text{L2 Regularization}
}
$$

The prediction equation is still:

$$
\hat{y}=w_1x_1+w_2x_2+\cdots+w_nx_n+b
$$

But the model learns the weights by minimizing:

$$
\boxed{
MSE+\alpha\sum w_j^2
}
$$

This penalty discourages unnecessarily large coefficients and can help the model generalize better to unseen data.
