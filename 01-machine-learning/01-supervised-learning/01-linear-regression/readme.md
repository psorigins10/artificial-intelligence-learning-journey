# Linear Regression

## What is Linear Regression?

**Linear Regression** is a supervised machine learning algorithm used to predict a **continuous numerical value**.

The basic idea is simple:

> Find a line that best describes the relationship between input `X` and output `y`.

For one feature, the line is written as:

$$
f(x) = mx + b
$$

Where:

* `x` → input feature
* `f(x)` → predicted output
* `m` → slope / coefficient
* `b` → intercept

---

## Simple Example

Suppose we want to predict a student's exam score based on how many hours they studied.

Our training data might look like:

| Hours Studied | Exam Score |
| ------------: | ---------: |
|             1 |         50 |
|             2 |         55 |
|             3 |         65 |
|             4 |         70 |
|             5 |         80 |

We want the model to learn a relationship between:

```text
X = Hours Studied
y = Exam Score
```

The model assumes a relationship like:

$$
f(x) = mx + b
$$

$$
Here, f(x) is pedicted value, m is the slope of the line and b is the intercept and x is the input.
$$

The values of `m` and `b` are **not chosen randomly by us**.

During training, the algorithm finds values of `m` and `b` that make the line fit the training data as well as possible.

---

# What are `m` and `b`?

Suppose the trained model finds:

$$
f(x) = 7x + 42
$$

Here:

```text
m = 7
b = 42
```

### `m` — Slope

`m` tells us how much the prediction changes when `x` increases by 1.

Here:

$$
m = 7
$$

So for every additional hour studied, the model predicts approximately **7 more marks**.

### `b` — Intercept

`b` is the predicted value when `x = 0`.

Here:

$$
b = 42
$$

So if a student studies 0 hours, the model predicts:

$$
f(0) = 7(0) + 42 = 42
$$

---

# Making a Prediction

Once the model has learned `m` and `b`, we can give it a new input.

Suppose:

```text
Hours studied = 6
```

The model is:

$$
f(x) = 7x + 42
$$

Therefore:

$$
f(6) = 7(6) + 42
$$

$$
f(6) = 84
$$

The model predicts an exam score of **84**.

---

# What Happens During Training?

This is the important part.

The model starts with the training data:

```text
Hours → Score

1 → 50
2 → 55
3 → 65
4 → 70
5 → 80
```

It tries to find the best values for:

```text
m → slope
b → intercept
```

This creates a line:

$$
\hat{y} = mx + b
$$

The model then compares its predictions with the actual values.

For example, suppose the model predicts:

```text
Actual score:    65
Predicted score: 63
```

The error is:

$$
65 - 63 = 2
$$

This difference is called the **residual** or **prediction error**.

# Ordinary Least Squares (OLS)

A common method used to find the best `m` and `b` is called **Ordinary Least Squares (OLS)**.

OLS tries to make the total squared prediction error as small as possible.

The objective is:

$$
\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

Where:

* `yᵢ` → actual value
* `ŷᵢ` → predicted value
* `yᵢ - ŷᵢ` → prediction error
* `n` → number of training examples

The errors are squared so that positive and negative errors don't cancel each other out.

For example:

```text
Errors:

+2
-3
+5
```

Without squaring:

```text
2 - 3 + 5 = 4
```

With squaring:

```text
2² + (-3)² + 5²
= 4 + 9 + 25
= 38
```

OLS chooses the line that minimizes this squared-error value.

---

# The Complete Process

```text
Training Data
      ↓
Linear Regression
      ↓
Find the best m and b
      ↓
Create the best-fit line
      ↓
Use the line for predictions
```

In simple terms:

> **Linear Regression finds a line.**
>
> **OLS finds the best values for the line's parameters.**

---

# Multiple Features

The simple formula uses one feature:

$$
f(x) = mx + b
$$

But real ML problems usually have multiple features.

For example, house price might depend on:

```text
x₁ = house size
x₂ = number of bedrooms
x₃ = age of house
```

The model becomes:

$$
f(x) = b + w_1x_1 + w_2x_2 + w_3x_3
$$

More generally:

$$
f(x) = b + \sum_{j=1}^{n} w_jx_j
$$

Here:

* `x₁, x₂, ...` → features
* `w₁, w₂, ...` → weights/coefficient values
* `b` → intercept
* `f(x)` → predicted value

The idea is still the same:

**Find the coefficients that make the predictions fit the training data.**

---

# Scikit-Learn

In scikit-learn, linear regression can be implemented using:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X, y)
```

After training:

```python
model.coef_
```

gives the learned coefficients (`m` or `w` values).

```python
model.intercept_
```

gives the learned intercept (`b`).

To make predictions:

```python
predictions = model.predict(X)
```

---

# Mental Model

Think of Linear Regression like this:

```text
             TRAINING
                ↓
        ┌────────────────┐
        │  Input X, y    │
        └───────┬────────┘
                ↓
        Find coefficients
          m / w₁ / w₂...
                +
              b
                ↓
        ┌────────────────┐
        │  Best-fit line │
        └───────┬────────┘
                ↓
             PREDICT
                ↓
          New X → ŷ
```

---

# Key Takeaways

* **Linear Regression** predicts continuous numerical values.

* The simplest form is:

  $$f(x) = mx + b$$

* `m` = slope / coefficient.

* `b` = intercept.

* During training, the model learns the best values for these parameters.

* **OLS (Ordinary Least Squares)** finds parameters by minimizing the sum of squared errors.

* After training, the model uses the learned equation to make predictions.

* With multiple features, the model becomes a weighted sum of the features.

### One-line summary

> **Linear Regression learns the best line from data so that it can predict new numerical values.**
