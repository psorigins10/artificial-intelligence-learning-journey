# Model Complexity

**Model complexity** describes how flexible a machine learning model is and how much detail it can learn from the training data.

A simple model has limited flexibility, while a complex model can learn more complicated patterns.

Model complexity is **not a metric like Accuracy, MSE, or F1 Score**. It is a concept used to understand the relationship between a model's flexibility and its ability to generalize to new data.

---

## Simple vs Complex Models

A simple model might look like:

```text
y = wx + b
```

This is a straight-line relationship.

A more complex model might contain many parameters or follow a complicated curve.

For example, polynomial regression:

```text
Degree 1 → Simple
Degree 2 → More complex
Degree 5 → More complex
Degree 9 → Very complex
```

As complexity increases, the model becomes more flexible.

---

## Underfitting and Overfitting

Model complexity is closely related to **underfitting** and **overfitting**.

### Underfitting

A model is too simple to learn the important patterns in the data.

```text
Too Simple
    ↓
Cannot learn enough
    ↓
High Training Error
    ↓
High Testing Error
```

### Good Complexity

The model is complex enough to learn the important patterns without memorizing the training data.

```text
Good Complexity
      ↓
Learns useful patterns
      ↓
Low Testing Error
      ↓
Good Generalization
```

### Overfitting

A model is too complex and starts learning noise or specific details from the training data.

```text
Too Complex
    ↓
Memorizes Training Data
    ↓
Very Low Training Error
    ↓
High Testing Error
```

---

## Polynomial Regression Example

Polynomial regression is useful for demonstrating model complexity because the polynomial degree controls how flexible the model is.

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Dataset
X = np.array([
    1, 2, 3, 4, 5,
    6, 7, 8, 9, 10
]).reshape(-1, 1)

y = np.array([
    2, 4, 5, 4, 7,
    8, 9, 11, 10, 14
])


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


# Different model complexities
degrees = [1, 2, 5, 9]

for degree in degrees:

    # Create model
    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Calculate errors
    train_error = mean_squared_error(y_train, train_pred)
    test_error = mean_squared_error(y_test, test_pred)

    print("Polynomial Degree:", degree)
    print("Training MSE:", train_error)
    print("Testing MSE:", test_error)
    print()
```

---

## Understanding the Code

The most important part is:

```python
degrees = [1, 2, 5, 9]
```

The polynomial degree controls the complexity of the model.

```text
Degree 1
   ↓
Simple model

Degree 2
   ↓
More complex

Degree 5
   ↓
Very flexible

Degree 9
   ↓
Extremely flexible
```

The code trains a different model for each degree.

It then calculates:

```text
Training MSE
Testing MSE
```

This lets us see how the model behaves as complexity increases.

---

## Training Error vs Testing Error

A typical pattern looks like:

```text
Model        Training Error    Testing Error
------------------------------------------------
Degree 1          High             High
Degree 2          Low              Low
Degree 5          Very Low         Higher
Degree 9          ~0               Very High
```

The exact numbers depend on the dataset and train/test split.

The important idea is the relationship between the two errors.

### Underfitting

```text
Training Error → High
Testing Error  → High
```

The model is too simple.

### Good Fit

```text
Training Error → Low
Testing Error  → Low
```

The model learns useful patterns and generalizes well.

### Overfitting

```text
Training Error → Very Low
Testing Error  → High
```

The model is too complex and is fitting the training data too closely.

---

## Why Testing Error Matters

A model's goal is not simply to perform well on the training data.

The goal is to perform well on **new, unseen data**.

For example:

```text
Training Data
     ↓
   Model
     ↓
Learns Pattern
     ↓
New Data
     ↓
Prediction
```

If the model only memorizes the training examples, it will perform poorly on new data.

This is why we compare training performance with testing performance.

---

## Complexity and Generalization

The goal is not:

```text
"Make the model as complex as possible."
```

The goal is:

```text
Find a model complex enough to learn
the real pattern but simple enough
to generalize to new data.
```

This is the idea behind choosing an appropriate model complexity.

---

## Visual Concept

The relationship can be thought of as:

```text
Error
  │
  │\                    Testing Error
  │ \                  /
  │  \                /
  │   \____      ____/
  │        \____/
  │
  │ Training Error
  │    \________________
  │
  └──────────────────────────→
       Model Complexity
```

As complexity increases:

- Training error generally decreases.
- Testing error may initially decrease.
- After a certain point, testing error starts increasing because of overfitting.

The lowest point of the testing error represents a useful level of model complexity.

---

## Key Takeaway

Model complexity describes **how flexible a model is**.

```text
Too Simple
    ↓
Underfitting

Good Complexity
    ↓
Good Generalization

Too Complex
    ↓
Overfitting
```

Remember:

```text
Model Complexity
       ↓
Training Error
       ↓
Testing Error
       ↓
Generalization
```

The objective is not to minimize training error alone.

The objective is to build a model that performs well on **data it has never seen before**.
