# Root Mean Squared Error (RMSE)

Root Mean Squared Error (RMSE) is a regression metric that measures the typical size of prediction errors while giving more weight to larger errors.

## Formula

$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$

RMSE is simply the **square root of MSE**:

$$
RMSE = \sqrt{MSE}
$$

Where:

* (y_i) = actual value
* (\hat{y}_i) = predicted value
* (n) = number of data points

## How RMSE Works

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
        ↓
   Square Root
        ↓
      RMSE
```

## Example

```python
import numpy as np

y = np.array([10, 20, 30, 40, 50])
y_pred = np.array([12, 18, 33, 37, 55])

errors = y - y_pred
squared_errors = errors ** 2

mse = np.mean(squared_errors)
rmse = np.sqrt(mse)

print(f"RMSE: {rmse}")
```

Output:

```text
RMSE: 3.492849839314596
```

## Why Use RMSE?

MSE is expressed in squared units.

For example, if the target is measured in kilograms:

```text
MSE  → kg²
RMSE → kg
```

Taking the square root brings the metric back to the **same units as the target variable**, making RMSE easier to interpret.

## RMSE vs MSE

Both metrics penalize large errors because they use squared errors.

The difference is:

* **MSE** stays in squared units.
* **RMSE** takes the square root and returns to the original units.

For example:

$$
MSE = 25
$$

$$
RMSE = \sqrt{25}=5
$$

## Interpreting RMSE

A lower RMSE means the model's predictions are generally closer to the actual values.

For example, if predicting house prices in ₹ lakh:

```text
RMSE = 3.5
```

means the model's prediction errors are on the scale of approximately **3.5 lakh**, with larger errors having more influence on the metric.

## RMSE with Scikit-Learn

Scikit-learn can calculate RMSE using `root_mean_squared_error()` in current versions:

```python
from sklearn.metrics import root_mean_squared_error

rmse = root_mean_squared_error(y, y_pred)

print(f"RMSE: {rmse}")
```

Alternatively, RMSE can always be calculated from MSE:

```python
from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
```

## Key Takeaway

> **RMSE is the square root of the average squared prediction errors.**

It is useful because it keeps the same units as the target variable while still giving larger errors more weight.
