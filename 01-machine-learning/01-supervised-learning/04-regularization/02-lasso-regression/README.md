# Lasso Regression

Lasso Regression is a regularized version of Linear Regression.

The main purpose of Lasso is to:

1. Reduce overfitting.
2. Keep model coefficients small.
3. Perform automatic feature selection by making some coefficients exactly zero.

---

## 1. Linear Regression First

Ordinary Linear Regression tries to find coefficients that minimize the Mean Squared Error (MSE).

For multiple features:

```text
y_pred = b + w1*x1 + w2*x2 + ... + wn*xn
```

Where:

- `y_pred` = predicted value
- `b` = intercept
- `w1, w2, ... wn` = coefficients (weights)
- `x1, x2, ... xn` = features

For example, a house-price model might be:

```text
price = b
      + w1 * area_sqft
      + w2 * bedrooms
      + w3 * bathrooms
      + w4 * parking
```

Linear Regression tries to find the weights that produce the lowest prediction error.

Its objective is:

```text
MSE
```

---

# 2. The Problem: Overfitting

Suppose we have many features:

```text
area_sqft
bedrooms
bathrooms
age
parking
floor
balcony
garden
renovated
...
```

Some features may not provide much useful information.

If the model tries too hard to fit the training data, it can learn noise instead of useful patterns.

This is called **overfitting**.

Regularization adds a penalty to discourage unnecessarily large coefficients.

---

# 3. What Lasso Changes

Lasso changes the Linear Regression objective from:

```text
MSE
```

to:

```text
MSE + alpha * sum(|w|)
```

Mathematically:

```text
J(w) = MSE + alpha * Σ|wj|
```

The important part is:

```text
alpha * Σ|wj|
```

This is called the **L1 penalty**.

---

# 4. What is `w`?

`w` is simply a coefficient (weight) learned by the model.

For example:

```text
area_sqft       → w1
bedrooms        → w2
bathrooms       → w3
parking         → w4
```

If the model learns:

```text
area_sqft  → 0.08
bedrooms   → 2.40
bathrooms  → 5.10
parking    → 0
```

then the parking feature has a coefficient of zero.

That means parking is not contributing to the model's prediction.

---

# 5. Why Does Lasso Use `|w|`?

Lasso uses:

```text
|w|
```

instead of:

```text
w²
```

The absolute-value penalty has a sharp point at zero.

This makes it possible for the optimization process to produce:

```text
w = 0
```

So Lasso does two things:

```text
1. Shrinks coefficients
2. Can make coefficients exactly zero
```

When a coefficient becomes zero, that feature is effectively removed from the model.

This is why Lasso is useful for **feature selection**.

---

# 6. Ridge vs Lasso

Ridge uses an L2 penalty:

```text
MSE + alpha * Σ(wj²)
```

Lasso uses an L1 penalty:

```text
MSE + alpha * Σ|wj|
```

### Ridge

Ridge usually shrinks coefficients:

```text
10.0 → 7.2
 2.0 → 1.4
 0.5 → 0.3
```

but normally does not make them exactly zero.

### Lasso

Lasso can shrink coefficients all the way to zero:

```text
10.0 → 7.0
 2.0 → 1.1
 0.5 → 0
```

Therefore:

```text
Ridge → shrink coefficients
Lasso → shrink + possible feature elimination
```

---

# 7. What Does `alpha` Mean?

`alpha` controls how strongly the model penalizes the coefficients.

### Small alpha

```text
alpha = 0.01
```

The penalty is weak.

The model behaves more like ordinary Linear Regression.

### Large alpha

```text
alpha = 10
```

The penalty is strong.

The model has more pressure to keep coefficients small.

For Lasso, a larger `alpha` can result in more coefficients becoming zero.

Conceptually:

```text
alpha increases
      ↓
penalty increases
      ↓
coefficients shrink more
      ↓
Lasso may eliminate more features
```

However, a very large `alpha` can make the model too simple and cause **underfitting**.

---

# 8. How Does Lasso Find the Weights?

Lasso does not simply look at a feature and decide:

> "This feature is irrelevant."

Instead, it tries to find coefficient values that minimize:

```text
MSE + alpha * Σ|w|
```

The optimization process starts with coefficient values and repeatedly updates them to reduce the objective.

Conceptually:

```text
Start with weights
       ↓
Calculate prediction error
       ↓
Calculate L1 penalty
       ↓
Calculate total cost
       ↓
Update weights
       ↓
Repeat
       ↓
Find weights with a low total cost
```

Scikit-learn's Lasso implementation commonly uses **coordinate descent** to solve this optimization problem.

---

# 9. Why Can Lasso Eliminate a Feature?

Imagine two possible choices.

### Keep the feature

```text
w = 0.05
```

This slightly improves the predictions, but it also adds an L1 penalty.

### Remove the feature

```text
w = 0
```

The feature contributes nothing to the prediction, but it also has no penalty.

If the improvement in MSE is not worth the additional penalty, the optimal solution can be:

```text
w = 0
```

That is how Lasso performs automatic feature selection.

---

# 10. Example with House Prices

Suppose we have:

```text
area_sqft
bedrooms
bathrooms
parking
balcony
garden
```

After training, Lasso might produce:

```text
area_sqft   → 0.05
bedrooms    → 2.10
bathrooms   → 4.80
parking     → 0
balcony     → 0
garden      → 1.20
```

The model has not "understood" what a balcony or parking space is.

It only sees numerical data.

For example:

```text
balcony = 0 → no balcony
balcony = 1 → balcony
```

Lasso looks at whether including that feature with a non-zero coefficient improves the objective enough.

If not, it can push the coefficient to zero.

Important:

```text
coefficient = 0
```

does **not** prove that the feature has no real-world effect.

It only means that, given the training data, other features, and chosen `alpha`, the Lasso model found that a zero coefficient was preferable.

---

# 11. Basic Python Example

```python
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split

df = pd.read_csv("house_price_lasso_practice.csv")

X = df.drop(["price_lakh"], axis=1)
y = df["price_lakh"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

reg = linear_model.Lasso(alpha=1)

reg.fit(X_train, y_train)

print("Intercept:", reg.intercept_)

for feature, coefficient in zip(X.columns, reg.coef_):
    print(f"{feature}: {coefficient:.4f}")
```

To see which features Lasso selected:

```python
for feature, coefficient in zip(X.columns, reg.coef_):
    if coefficient == 0:
        print(f"{feature} was eliminated")
```

---

# 12. Important: Scale Your Features

Lasso is sensitive to the scale of features because the penalty is applied directly to the coefficients.

For example:

```text
area_sqft       → values around 3000
bedrooms        → values around 3
balcony         → values 0 or 1
```

These features are on very different scales.

In a proper Lasso workflow, feature scaling is usually important.

A common approach is:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

model = make_pipeline(
    StandardScaler(),
    Lasso(alpha=0.1)
)

model.fit(X_train, y_train)
```

Scaling ensures that the regularization penalty treats features more fairly.

---

# 13. When Should You Use Lasso?

Lasso is useful when:

- You have many features.
- You suspect some features may be irrelevant.
- You want automatic feature selection.
- You want to reduce overfitting.
- You want a simpler model with fewer active features.

It is especially useful when you believe that only a subset of your features is genuinely important.

---

# 14. Linear Regression vs Ridge vs Lasso

```text
Linear Regression
        ↓
Minimize MSE
        ↓
No regularization


Ridge Regression
        ↓
Minimize MSE + alpha * Σ(w²)
        ↓
Shrink coefficients


Lasso Regression
        ↓
Minimize MSE + alpha * Σ|w|
        ↓
Shrink coefficients
        ↓
Can make coefficients exactly zero
        ↓
Automatic feature selection
```

---

# 15. Key Takeaways

The most important things to remember:

```text
Linear Regression:
MSE
```

```text
Ridge:
MSE + alpha * Σ(w²)
```

```text
Lasso:
MSE + alpha * Σ|w|
```

`w` = coefficient/weight.

`alpha` = regularization strength.

Lasso uses an **L1 penalty**.

Ridge uses an **L2 penalty**.

Lasso can make coefficients exactly zero.

Therefore:

```text
Lasso = regularization + feature selection
```

And remember: a coefficient becoming zero does not automatically mean the feature is objectively useless. It means the trained model found that removing it gave the best trade-off between prediction error and the L1 penalty for that particular dataset and `alpha`.
