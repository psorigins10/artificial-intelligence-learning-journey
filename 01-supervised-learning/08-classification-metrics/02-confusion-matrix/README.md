# Confusion Matrix

A **Confusion Matrix** is a table used to evaluate the performance of a **classification model**.

It shows not only how many predictions were correct, but also **what kind of mistakes the model made**.

For binary classification, there are four possible results:

- True Positive (TP)
- True Negative (TN)
- False Positive (FP)
- False Negative (FN)

---

## Basic Structure

For a binary classification problem:

```text
                 Predicted
                 Negative  Positive

Actual Negative    TN        FP
       Positive    FN        TP
```

Where:

### True Positive (TP)

The actual class is positive and the model predicts positive.

```text
Actual:    Positive
Predicted: Positive
```

The model is correct.

### True Negative (TN)

The actual class is negative and the model predicts negative.

```text
Actual:    Negative
Predicted: Negative
```

The model is correct.

### False Positive (FP)

The actual class is negative but the model predicts positive.

```text
Actual:    Negative
Predicted: Positive
```

The model is wrong.

### False Negative (FN)

The actual class is positive but the model predicts negative.

```text
Actual:    Positive
Predicted: Negative
```

The model is wrong.

---

## Pass / Fail Example

Suppose:

```text
0 = Fail
1 = Pass
```

A confusion matrix could look like:

```text
                 Predicted
                 Fail   Pass

Actual Fail        4      1
Actual Pass        1      4
```

Therefore:

```text
TN = 4
FP = 1
FN = 1
TP = 4
```

This means:

- 4 students actually failed and were correctly predicted as Fail.
- 4 students actually passed and were correctly predicted as Pass.
- 1 student actually failed but was predicted as Pass.
- 1 student actually passed but was predicted as Fail.

---

## Creating a Confusion Matrix with Scikit-learn

Scikit-learn provides `confusion_matrix()`.

```python
import numpy as np

from sklearn.metrics import confusion_matrix


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


# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
```

Output:

```text
Confusion Matrix:
[[4 1]
 [1 4]]
```

---

## Getting TN, FP, FN, and TP

You can extract the four values using:

```python
TN, FP, FN, TP = cm.ravel()

print("True Negative:", TN)
print("False Positive:", FP)
print("False Negative:", FN)
print("True Positive:", TP)
```

Output:

```text
True Negative: 4
False Positive: 1
False Negative: 1
True Positive: 4
```

---

## Plotting the Confusion Matrix

Scikit-learn also provides `ConfusionMatrixDisplay`.

```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fail", "Pass"]
)

display.plot()

plt.title("Confusion Matrix")
plt.show()
```

This produces a visual version of the matrix.

---

## Confusion Matrix and Accuracy

Accuracy can be calculated from the confusion matrix:

$$
Accuracy =
rac{TP + TN}
{TP + TN + FP + FN}
$$

Using the example:

```text
TN = 4
FP = 1
FN = 1
TP = 4
```

Therefore:

$$
Accuracy =
rac{4 + 4}
{4 + 4 + 1 + 1}
$$

$$
Accuracy = rac{8}{10}
$$

$$
Accuracy = 0.8
$$

So:

```text
Accuracy = 80%
```

---

## Confusion Matrix and Other Metrics

The four values in a confusion matrix are the foundation for several classification metrics.

### Accuracy

$$
Accuracy =
rac{TP + TN}
{TP + TN + FP + FN}
$$

Measures the overall percentage of correct predictions.

### Precision

$$
Precision =
rac{TP}
{TP + FP}
$$

Answers:

> When the model predicts positive, how often is it actually positive?

### Recall

$$
Recall =
rac{TP}
{TP + FN}
$$

Answers:

> Out of all actual positive cases, how many did the model find?

### F1 Score

$$
F1 =
2 	imes
rac{Precision 	imes Recall}
{Precision + Recall}
$$

Combines Precision and Recall into one score.

---

## Why Confusion Matrix Is Useful

Accuracy only tells you:

```text
How many predictions were correct?
```

A confusion matrix tells you:

```text
How many were correctly positive?
How many were correctly negative?
How many false positives?
How many false negatives?
```

This is important because different mistakes can have different consequences.

For example, in a disease detection model:

```text
False Positive:
Healthy → predicted Sick

False Negative:
Sick → predicted Healthy
```

A false negative could be much more serious than a false positive.

Therefore, looking only at accuracy would hide important information.

---

## Key Takeaway

A confusion matrix breaks classification predictions into four categories:

```text
                Predicted
              Negative  Positive

Actual Negative    TN       FP
       Positive    FN       TP
```

Remember:

```text
TP → Correct Positive
TN → Correct Negative
FP → Wrong Positive
FN → Wrong Negative
```

These four values are then used to calculate:

```text
Confusion Matrix
       ↓
 ┌─────┼─────┐
 ↓     ↓     ↓
Accuracy  Precision  Recall
              \      /
               \    /
                F1
```

A confusion matrix is one of the most important tools for understanding how a classification model is actually making mistakes.
