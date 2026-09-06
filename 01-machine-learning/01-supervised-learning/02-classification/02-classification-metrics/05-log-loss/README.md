# Log Loss

**Log Loss**, also called **Logarithmic Loss** or **Cross-Entropy Loss**, is a classification metric that measures how well a model's predicted probabilities match the actual classes.

It is commonly used with **Logistic Regression** and other classification models that produce probabilities.

---

## What Does Log Loss Measure?

Accuracy only checks whether the final prediction is correct or incorrect.

Log Loss looks at the **probability behind the prediction**.

For example:

```text
Actual = Pass

Model A → 0.90 probability of Pass
Model B → 0.55 probability of Pass
Model C → 0.10 probability of Pass
```

All three can be converted to a Pass prediction using a `0.5` threshold, but they are not equally good predictions.

Log Loss gives:

- A small penalty to confident correct predictions.
- A larger penalty to uncertain predictions.
- A very large penalty to confident incorrect predictions.

**Lower Log Loss is better.**

---

## Binary Classification Formula

For one example:

$$
LogLoss = -[y\\log(p) + (1-y)\\log(1-p)]
$$

Where:

- `y` = actual class (`0` or `1`)
- `p` = predicted probability of class `1`

For multiple examples, the losses are averaged:

$$
LogLoss = -\\frac{1}{N} \\sum_{i=1}^{N}[y_i\\log(p_i) + (1-y_i)\\log(1-p_i)]
$$

---

## Simple Example

Suppose:

```text
Actual = 1
Predicted probability = 0.9
```

The formula becomes:

$$
LogLoss = -\\log(0.9)
$$

Approximately:

$$
LogLoss \\approx 0.105
$$

This is a small loss because the model was confident and correct.

Now suppose:

```text
Actual = 1
Predicted probability = 0.1
```

Then:

$$
LogLoss = -\\log(0.1)
$$

Approximately:

$$
LogLoss \\approx 2.303
$$

The loss is much larger because the model was confidently wrong.

---

## Using Scikit-learn

```python
import numpy as np
from sklearn.metrics import log_loss


y_true = np.array([1, 1, 0, 0, 1])

y_probability = np.array([0.9, 0.8, 0.2, 0.1, 0.7])

loss = log_loss(y_true, y_probability)

print("Log Loss:", loss)
```

---

## Log Loss with Logistic Regression

```text
Input X
   ↓
z = wx + b
   ↓
Sigmoid
   ↓
Probability
   ↓
Log Loss
```

The model tries to learn parameters that produce good probabilities and therefore a low loss.

---

## Log Loss vs Accuracy

Accuracy asks:

```text
Did the model get the class correct?
```

Log Loss asks:

```text
How good was the probability prediction?
```

For example:

```text
Actual = 1

Model A → 0.51
Model B → 0.99
```

Both predict class `1` using a `0.5` threshold, but Log Loss prefers Model B because it was much more confident and still correct.

---

## Key Takeaway

Log Loss measures how good a classification model's **predicted probabilities** are.

```text
LOW Log Loss  = GOOD
HIGH Log Loss = BAD
```

The most important rule is:

```text
Confident + Correct
        ↓
   Small Loss

Confident + Wrong
        ↓
   Large Loss
```
