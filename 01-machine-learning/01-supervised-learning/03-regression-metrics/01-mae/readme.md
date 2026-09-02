# Mean Absolute Error (MAE)

Mean Absolute Error (MAE) is a regression metric used to measure the average absolute difference between a model's predictions and the actual values.

## Formula

$$
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

Where:

* (y_i) = actual value
* (\hat{y}_i) = predicted value
* (n) = number of data points

## How MAE Works

For every prediction:

1. Calculate the error between the actual and predicted value.
2. Take the absolute value of the error.
3. Calculate the mean of all absolute errors.

```text
Actual - Predicted
        ↓
      Error
        ↓
  Absolute Value
        ↓
      Mean
        ↓
      MAE
```

### Example

```python
import numpy as np

y = np.array([10, 20, 30, 40, 50])
y_pred = np.array([12, 18, 33, 37, 55])

errors = y - y_pred
absolute_errors = np.abs(errors)

mae = np.mean(absolute_errors)

print(f"MAE: {mae}")
```

Output:

```text
MAE: 3.0
```

## Why Use Absolute Value?

Prediction errors can be positive or negative.

For example:

```text
Actual:    10
Predicted: 12
Error:     -2

Actual:    20
Predicted: 18
Error:      2
```

If we simply averaged the errors:

$$
\frac{-2+2}{2}=0
$$

It would incorrectly suggest that there was no error.

Taking the absolute value prevents positive and negative errors from cancelling each other:

$$
|-2|=2
$$

$$
|2|=2
$$

## MAE vs MSE

The main difference is how they treat errors.

### MAE

$$
MAE = \frac{1}{n}\sum|y-\hat{y}|
$$

MAE treats errors linearly.

### MSE

$$
MSE = \frac{1}{n}\sum(y-\hat{y})^2
$$

MSE heavily penalizes large errors because they are squared.

For example:

```text
Error       MAE contribution       MSE contribution

2                 2                       4
10               10                     100
20               20                     400
```

Therefore, MAE is less sensitive to large errors than MSE.

## Interpreting MAE

MAE is easy to interpret because it uses the **same units as the target variable**.

For example, if you are predicting house prices in ₹ lakh and:

```text
MAE = 4.2
```

then the model's predictions are off by about **4.2 lakh on average** in absolute terms.

A lower MAE means the model's predictions are closer to the actual values.

## MAE with Scikit-Learn

Scikit-learn provides `mean_absolute_error()`:

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y, y_pred)

print(f"MAE: {mae}")
```

This produces the same result as the manual NumPy calculation.

## Key Takeaway

> **MAE is the average of the absolute differences between actual and predicted values.**

MAE is useful when you want an easy-to-understand measure of the typical prediction error without giving extremely large errors disproportionate influence.
