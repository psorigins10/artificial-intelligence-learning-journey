# Accuracy Score

Accuracy Score is a classification metric that measures how many predictions a model got correct out of all predictions.

## Formula

$$
Accuracy = \frac{Correct\ Predictions}{Total\ Predictions}
$$

Using a confusion matrix:

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

Where:

- `TP` = True Positive
- `TN` = True Negative
- `FP` = False Positive
- `FN` = False Negative

---

## Simple Example

Suppose a model makes 10 predictions:

```text
Actual:
[0, 1, 1, 1, 0, 1, 1, 1, 1, 0]

Predicted:
[0, 1, 1, 0, 0, 1, 1, 1, 0, 0]
```

The model correctly predicts 7 out of 10 examples.

$$
Accuracy = \frac{7}{10}
$$

$$
Accuracy = 0.7
$$

So:

```text
Accuracy = 70%
```

---

## Using Scikit-learn

Scikit-learn provides `accuracy_score()` for calculating accuracy.

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
```

Example output:

```text
Accuracy: 0.7
```

To display it as a percentage:

```python
print("Accuracy:", accuracy * 100, "%")
```

Output:

```text
Accuracy: 70.0 %
```

---

## Example with Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Predictions:", y_pred)
print("Accuracy:", accuracy)
```

---

## Accuracy and Confusion Matrix

Accuracy can also be calculated from a confusion matrix.

Example:

```text
                 Predicted
                 0      1

Actual  0        3      1
        1        1      5
```

Here:

```text
TN = 3
FP = 1
FN = 1
TP = 5
```

Therefore:

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

$$
Accuracy = \frac{5 + 3}{5 + 3 + 1 + 1}
$$

$$
Accuracy = \frac{8}{10}
$$

$$
Accuracy = 0.8
$$

Therefore:

```text
Accuracy = 80%
```

---

## When Accuracy Works Well

Accuracy works well when the classes are reasonably balanced.

For example:

```text
100 students

50 → Pass
50 → Fail
```

If the model predicts 90 correctly:

```text
Accuracy = 90%
```

This gives a useful representation of the model's performance.

---

## When Accuracy Can Be Misleading

Accuracy can be misleading when the dataset is imbalanced.

For example:

```text
1000 patients

990 → Healthy
10  → Sick
```

Imagine a model that predicts:

```text
Everyone → Healthy
```

It gets:

```text
990 / 1000 = 99%
```

accuracy.

But it detects:

```text
0 / 10 = 0%
```

of the sick patients.

So despite having 99% accuracy, the model is not useful for detecting disease.

This is why we also use:

- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Accuracy vs Other Classification Metrics

| Metric | What it measures |
|---|---|
| Accuracy | Overall percentage of correct predictions |
| Precision | How many predicted positives were actually positive |
| Recall | How many actual positives were correctly found |
| F1 Score | Balance between Precision and Recall |
| Confusion Matrix | Shows TP, TN, FP, and FN |

---

## Key Takeaway

Accuracy answers one simple question:

> **"Out of all predictions, how many were correct?"**

The formula is:

$$
\boxed{
Accuracy =
\frac{TP+TN}
{TP+TN+FP+FN}
}
$$

Remember:

```text
Correct Predictions
        ÷
Total Predictions
        =
     Accuracy
```

Accuracy is useful, but it should not be used alone when the classes are heavily imbalanced.
