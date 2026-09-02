# F1 Score

F1 Score is a **classification metric** that combines **Precision** and **Recall** into a single score.

It is especially useful when the classes are imbalanced or when both false positives and false negatives are important.

---

## Formula

$$
F1 = 2 \times
\frac{Precision \times Recall}
{Precision + Recall}
$$

F1 Score ranges from:

```text
0 → Worst
1 → Perfect
```

---

## Precision and Recall

F1 Score is built from two other classification metrics.

### Precision

Precision answers:

> When the model predicts positive, how often is it actually positive?

$$
Precision =
\frac{TP}
{TP + FP}
$$

### Recall

Recall answers:

> Out of all actual positive cases, how many did the model correctly find?

$$
Recall =
\frac{TP}
{TP + FN}
$$

---

## Simple Example

Suppose:

```text
Precision = 0.8
Recall = 0.6
```

Then:

$$
F1 =
2 \times
\frac{0.8 \times 0.6}
{0.8 + 0.6}
$$

$$
F1 =
2 \times
\frac{0.48}{1.4}
$$

$$
F1 \approx 0.686
$$

Therefore:

```text
F1 Score ≈ 68.6%
```

---

## Why Not Just Use Accuracy?

Accuracy only tells us how many predictions were correct overall.

For example:

```text
1000 examples

990 → Negative
10  → Positive
```

A model could predict every example as Negative and get:

```text
Accuracy = 99%
```

But it would completely fail to identify the positive cases.

Precision and Recall give us more information about the model's classification performance.

F1 combines both into one score.

---

## Using Scikit-learn

Scikit-learn provides `f1_score()` for calculating F1 Score.

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred)

print("F1 Score:", f1)
```

---

## Complete Demo

```python
import numpy as np

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# Actual values
y_test = np.array([
    0, 0, 0, 1, 1,
    1, 1, 0, 1, 0
])

# Model predictions
y_pred = np.array([
    0, 0, 1, 1, 1,
    0, 1, 0, 1, 0
])


# Calculate Precision
precision = precision_score(y_test, y_pred)

# Calculate Recall
recall = recall_score(y_test, y_pred)

# Calculate F1 Score
f1 = f1_score(y_test, y_pred)


print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
```

Example output:

```text
Precision: 0.8
Recall: 0.8
F1 Score: 0.8
```

---

## Calculating F1 Manually

You can also calculate F1 without Scikit-learn once you already have Precision and Recall.

```python
precision = 0.8
recall = 0.8

f1 = 2 * (precision * recall) / (precision + recall)

print("F1 Score:", f1)
```

Output:

```text
F1 Score: 0.8
```

---

## Connection to the Confusion Matrix

F1 Score ultimately depends on the values from the confusion matrix:

```text
                 Predicted
                 Negative  Positive

Actual Negative    TN        FP
       Positive    FN        TP
```

These values are used to calculate:

```text
          Confusion Matrix
                 ↓
          TP, TN, FP, FN
                 ↓
       ┌─────────┴─────────┐
       ↓                   ↓
   Precision             Recall
       └─────────┬─────────┘
                 ↓
              F1 Score
```

---

## When F1 Score Is Useful

F1 Score is useful when:

- The dataset has imbalanced classes.
- False positives matter.
- False negatives matter.
- You want one metric that considers both Precision and Recall.

For example, in a disease detection system, both types of mistakes can be important:

```text
False Positive:
Healthy → predicted Sick

False Negative:
Sick → predicted Healthy
```

F1 helps evaluate the balance between these types of errors.

---

## Key Takeaway

F1 Score is the **harmonic mean of Precision and Recall**.

$$
\boxed{
F1 = 2 \times
\frac{Precision \times Recall}
{Precision + Recall}
}
$$

Remember:

```text
Precision → How reliable are positive predictions?

Recall → How many actual positives did we find?

F1 → How well are Precision and Recall balanced?
```

A high F1 Score generally means the model has both good Precision and good Recall.
