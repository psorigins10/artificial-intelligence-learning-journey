# Mean Squared Error (MSE)

Mean Squared Error (MSE) is a regression metric used to measure how far a model's predictions are from the actual values.

## Formula

$$
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

Where:

* (y_i) = actual value
* (\hat{y}_i) = predicted value
* (n) = number of data points

## How MSE Works

For every prediction:

1. Calculate the error between the actual and predicted value.
2. Square the error.
3. Calculate the mean of all squared errors.

```text
Actual - Predicted
        ↓
      Error
        ↓
   Square Error
        ↓
      Mean
        ↓
       MSE
```

### Example

```python
import numpy as np

y = np.array([10, 20, 30, 40, 50])
y_pred = np.array([12, 18, 33, 37, 55])

errors = y - y_pred
squared_errors = errors ** 2

mse = np.mean(squared_errors)

print(f"MSE: {mse}")
```

Output:

```text
MSE: 12.2
```

## Why Square the Errors?

Squaring the errors does two important things:

* It removes negative values.
* It gives much more weight to large errors.

For example:

```text
Error:        -2    →    Squared: 4
Error:        -10   →    Squared: 100
```

Therefore, a large prediction error has a much bigger effect on MSE.

## Interpreting MSE

A **lower MSE means the model's predictions are generally closer to the actual values**.

```text
MSE = 0
   ↓
Perfect predictions

Lower MSE
   ↓
Better model
```

However, there is no universal value that determines whether an MSE is "good" or "bad." It depends on the scale of the target variable and the problem.

MSE is also expressed in **squared units** of the target variable. For this reason, RMSE is often used when an error value in the original units is easier to interpret.

## MSE with Scikit-Learn

Scikit-learn provides `mean_squared_error()`:

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y, y_pred)

print(f"MSE: {mse}")
```

This produces the same result as the manual NumPy calculation.

## Key Takeaway

> **MSE is the average of the squared differences between actual and predicted values.**

The lower the MSE, the smaller the model's squared prediction errors.
