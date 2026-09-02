# Polynomial Regression

Polynomial Regression is a regression technique used to model **nonlinear relationships** between input features and a target.

## Why Polynomial Regression?

Linear Regression assumes the relationship between `X` and `y` is a straight line:

$$
y = b + w_1x
$$

But real data can have curved relationships.

Polynomial Regression solves this by adding powers of the original feature:

$$
y = b + w_1x + w_2x^2 + w_3x^3 + \cdots
$$

For example, a degree-2 polynomial produces:

$$
y = b + w_1x + w_2x^2
$$

This allows Linear Regression to fit a **parabolic curve**.

## How it works

Polynomial Regression does not use a completely different regression algorithm.

It first transforms the features:

```text
x
↓
[1, x, x²]
↓
Linear Regression
↓
Prediction
```

In Scikit-learn, this can be done with:

```python
PolynomialFeatures(degree=2)
```

combined with:

```python
LinearRegression()
```

## Polynomial Degree

The degree determines how complex the model can be.

* Degree 1 → straight line
* Degree 2 → parabola
* Degree 3 → more flexible curve
* Higher degrees → increasingly complex curves

Higher degree does **not** automatically mean better performance. Very high degrees can cause **overfitting**, where the model fits the training data too closely and performs poorly on unseen data.

## Example

For a degree-2 model:

$$
y = b + w_1x + w_2x^2
$$

The model learns the values of `b`, `w₁`, and `w₂` from the training data.

## Key Idea

**Polynomial Regression extends Linear Regression by transforming the input features into polynomial features, allowing a linear model to represent nonlinear relationships.**
