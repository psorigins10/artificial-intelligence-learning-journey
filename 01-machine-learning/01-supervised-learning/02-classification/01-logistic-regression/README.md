# Logistic Regression

Logistic Regression is a supervised machine learning algorithm mainly used for **classification problems**.

Unlike Linear Regression, which predicts continuous numerical values, Logistic Regression predicts the **probability of a class**.

For example:

- `0` → Fail
- `1` → Pass

A simple example is predicting whether a student passes based on the number of hours they studied.

---

## How Logistic Regression Works

The basic flow is:

```text
Input X
   ↓
Linear equation
   ↓
z = wx + b
   ↓
Sigmoid function
   ↓
Probability
   ↓
Threshold
   ↓
Class 0 or 1
```

---

## 1. Linear Combination

Logistic Regression first calculates a linear value:

$$
z = wx + b
$$

Where:

- `x` = input feature
- `w` = weight
- `b` = bias
- `z` = linear score

For multiple features:

$$
z = w_1x_1 + w_2x_2 + ... + w_nx_n + b
$$

This is similar to the equation used in Linear Regression.

However, Logistic Regression does **not** use `z` directly as the final prediction.

---

## 2. Sigmoid Function

The linear value `z` is passed through the sigmoid function:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

The sigmoid function converts any value of `z` into a value between `0` and `1`.

```text
z → -∞       sigmoid(z) → 0

z → 0        sigmoid(z) → 0.5

z → +∞       sigmoid(z) → 1
```

The value produced by the sigmoid is interpreted as a probability.

For example:

```text
0.10 → 10% probability
0.50 → 50% probability
0.85 → 85% probability
```

---

## What is e?

`e` is Euler's number, a mathematical constant:

$$
e \approx 2.71828
$$

It appears in the sigmoid function:

$$
\frac{1}{1+e^{-z}}
$$

It is not a special variable created by Logistic Regression.

---

## 3. Probability

After applying the sigmoid function:

$$
P(y=1|x) = \frac{1}{1+e^{-(wx+b)}}
$$

This gives the probability that the input belongs to class `1`.

For example:

```text
Hours studied = 2

P(Pass) = 0.08
```

The model thinks there is an 8% probability of passing.

Another student:

```text
Hours studied = 5

P(Pass) = 0.96
```

The model thinks there is a 96% probability of passing.

---

## 4. Classification Threshold

The probability is converted into a class using a threshold.

The standard threshold is:

$$
0.5
$$

The rule is:

```text
Probability < 0.5  → Class 0
Probability ≥ 0.5  → Class 1
```

For the student example:

```text
Probability       Prediction

0.20              Fail (0)
0.35              Fail (0)
0.49              Fail (0)
0.50              Pass (1)
0.72              Pass (1)
0.95              Pass (1)
```

Therefore:

```text
0 = Fail
1 = Pass
```

---

## Complete Mathematical Process

The complete Logistic Regression calculation is:

### Step 1 — Linear Score

$$
z = wx+b
$$

### Step 2 — Sigmoid

$$
p = \frac{1}{1+e^{-z}}
$$

### Step 3 — Classification

$$
\hat{y} =
\begin{cases}
0 & p < 0.5 \\
1 & p \geq 0.5
\end{cases}
$$

So the complete process is:

```text
X
↓
wx + b
↓
z
↓
Sigmoid(z)
↓
Probability
↓
0.5 Threshold
↓
0 or 1
```

---

# Example

Suppose the model learns:

```text
w = 2
b = -7
```

A student studied:

```text
x = 4
```

First calculate `z`:

$$
z = wx+b
$$

$$
z = 2(4)-7
$$

$$
z = 1
$$

Now apply the sigmoid:

$$
p = \frac{1}{1+e^{-1}}
$$

$$
p \approx 0.731
$$

So:

```text
Probability of Pass ≈ 73.1%
```

Since:

```text
0.731 > 0.5
```

The prediction is:

```text
Pass (1)
```

---

# How Does the Model Learn?

The model needs to find the best values for:

```text
w = weight
b = bias
```

During training, Logistic Regression:

1. Calculates `z = wx + b`
2. Calculates the probability using sigmoid
3. Calculates the error using **Log Loss**
4. Adjusts `w` and `b`
5. Repeats the process until the loss is minimized

Conceptually:

```text
Initial w and b
      ↓
   wx + b
      ↓
   Sigmoid
      ↓
 Probability
      ↓
  Log Loss
      ↓
Adjust w and b
      ↓
    Repeat
```

---

# Log Loss

Logistic Regression commonly uses **Log Loss**, also called **Binary Cross-Entropy**, for binary classification.

The formula is:

$$
L = -[y\log(p)+(1-y)\log(1-p)]
$$

Where:

- `y` = actual class
- `p` = predicted probability

The goal is to minimize Log Loss.

A confident correct prediction gets a small loss.

A confident incorrect prediction gets a large loss.

---

# Logistic Regression vs Linear Regression

| Linear Regression | Logistic Regression |
|---|---|
| Regression | Classification |
| Predicts continuous values | Predicts class probabilities |
| Output can be any real number | Output is between 0 and 1 |
| Common metrics: MSE, MAE, RMSE, R² | Common metrics: Log Loss, Accuracy, Precision, Recall, F1 |
| Example: house price | Example: Pass/Fail |
| Linear output | Sigmoid output |

Linear Regression:

$$
\hat{y}=wx+b
$$

Logistic Regression:

$$
p=\frac{1}{1+e^{-(wx+b)}}
$$

---

# Evaluation Metrics

For Logistic Regression classification, useful metrics include:

### Accuracy

Measures how many predictions were correct.

$$
Accuracy =
\frac{Correct\ Predictions}{Total\ Predictions}
$$

### Confusion Matrix

Shows:

- True Positive
- True Negative
- False Positive
- False Negative

### Precision

Measures how many predicted positives were actually positive.

### Recall

Measures how many actual positives were correctly identified.

### F1 Score

Combines Precision and Recall.

---

# Python Implementation

```python
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


X = np.array([
    0.5, 0.7, 0.9, 1.0, 1.2,
    1.4, 1.5, 1.7, 1.8, 2.0,
    2.1, 2.2, 2.4, 2.5, 2.6,
    2.8, 2.9, 3.0, 3.1, 3.2,
    3.3, 3.4, 3.5, 3.6, 3.7,
    3.8, 3.9, 4.0, 4.1, 4.2,
    4.3, 4.5, 4.6, 4.8, 5.0,
    5.2, 5.4, 5.5, 5.7, 6.0,
    6.2, 6.5, 6.8, 7.0, 7.2,
    7.5, 8.0, 8.2, 8.5, 9.0
])

y = np.array([
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1
])

# Reshape X
X = X.reshape(-1, 1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Predict classes
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
```

---

# Key Takeaway

The most important thing to remember about Logistic Regression is:

$$
X \rightarrow wx+b \rightarrow sigmoid \rightarrow probability \rightarrow threshold \rightarrow class
$$

It does **not** directly predict `0` or `1`.

It first calculates a probability and then uses a threshold to make the final classification.

```text
Input
  ↓
Linear score
z = wx + b
  ↓
Sigmoid
1 / (1 + e^-z)
  ↓
Probability
  ↓
Threshold
  ↓
Class
0 or 1
```
