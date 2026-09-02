# Train/Test Split & R² Score

## Train/Test Split

In machine learning, we split our dataset into two parts:

* **Training data** → used by the model to learn patterns.
* **Testing data** → used to evaluate the model on data it hasn't seen before.

Example:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Here, `test_size=0.2` means:

```text
80% → Training data
20% → Testing data
```

`random_state=42` makes the random split reproducible. It works like a seed: using the same value produces the same split.

## Why Split the Data?

If we train and test on the same data, the model could appear better than it actually is.

The test set gives us an idea of how well the model **generalizes to unseen data**.

---

## R² Score

R² (R-squared), also called the **coefficient of determination**, measures how well a regression model explains the variation in the target variable.

In scikit-learn:

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
```

Generally:

```text
R² = 1     → Perfect predictions
R² ≈ 0.9  → Very strong fit
R² ≈ 0.5  → Moderate fit
R² = 0     → No improvement over predicting the mean
R² < 0     → Model performs worse than the baseline
```

For example:

```text
R² = 0.9973
```

means the model explains approximately **99.73% of the variation** in the test data.

## Complete Workflow

```text
Dataset
   ↓
Train/Test Split
   ↓
Training Data → model.fit()
   ↓
Trained Model
   ↓
Test Data → model.predict()
   ↓
Compare Predictions with y_test
   ↓
R² Score
```

This is the basic workflow I used while learning linear regression with scikit-learn..
