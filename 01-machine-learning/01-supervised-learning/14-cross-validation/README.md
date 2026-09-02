# Cross-Validation

## What is Cross-Validation?

**Cross-Validation** is a technique used to estimate how well a machine learning model will perform on **unseen data**.

Instead of relying on only one train/validation split, cross-validation splits the dataset into multiple parts called **folds** and trains/evaluates the model multiple times.

The goal is to get a more reliable estimate of the model's performance.

---

## Why Do We Need Cross-Validation?

Suppose we have 1,000 samples.

A normal train/validation split might look like:

```text
Dataset
   │
   ├── Training Data → 800 samples
   │
   └── Validation Data → 200 samples
```

The problem is that the result can depend heavily on **which 200 samples** happened to be selected.

Cross-validation reduces this dependency by using different portions of the dataset for validation.

---

# K-Fold Cross-Validation

The most common type of cross-validation is **K-Fold Cross-Validation**.

If:

```python
cv = 5
```

the dataset is divided into 5 folds.

```text
┌────┬────┬────┬────┬────┐
│ F1 │ F2 │ F3 │ F4 │ F5 │
└────┴────┴────┴────┴────┘
```

The model is trained and evaluated 5 times.

### Fold 1

```text
Training: F2 F3 F4 F5
Validation: F1
```

### Fold 2

```text
Training: F1 F3 F4 F5
Validation: F2
```

### Fold 3

```text
Training: F1 F2 F4 F5
Validation: F3
```

### Fold 4

```text
Training: F1 F2 F3 F5
Validation: F4
```

### Fold 5

```text
Training: F1 F2 F3 F4
Validation: F5
```

Every sample gets used for:

* Training multiple times
* Validation exactly once

---

# Cross-Validation Workflow

```text
             Dataset
                │
                ▼
        Split into K folds
                │
                ▼
       ┌─────────────────┐
       │ Train on K-1    │
       │ Validate on 1   │
       └─────────────────┘
                │
                ▼
          Repeat K times
                │
                ▼
        Get K validation scores
                │
                ▼
        Calculate mean score
```

---

# Using Cross-Validation in Scikit-Learn

Scikit-learn provides the `cross_val_score()` function.

```python
from sklearn.model_selection import cross_val_score
```

Example:

```python
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.svm import SVC

X, y = make_classification(
    n_samples=200,
    n_features=5,
    random_state=0
)

clf = SVC(
    kernel="linear"
)

scores = cross_val_score(
    clf,
    X,
    y,
    cv=5
)

print(scores)
print(scores.mean())
```

Example output:

```text
[0.95 0.95 0.90 0.95 1.00]

0.95
```

The five values are the scores from the five folds.

---

# Understanding `cross_val_score()`

The basic structure is:

```python
cross_val_score(model, X, y, cv=5)
```

### `model`

The machine learning model you want to evaluate.

```python
clf = SVC()
```

### `X`

Your input features.

```python
X
```

### `y`

Your target values.

```python
y
```

### `cv`

The number of folds.

```python
cv=5
```

means 5-fold cross-validation.

---

# Does `cross_val_score()` Fit the Model?

Yes.

When you run:

```python
scores = cross_val_score(clf, X, y, cv=5)
```

Scikit-learn automatically performs the training and evaluation for every fold.

Conceptually:

```text
Fold 1
   ↓
fit()
   ↓
predict()
   ↓
calculate score

Fold 2
   ↓
fit()
   ↓
predict()
   ↓
calculate score

...

Fold 5
   ↓
fit()
   ↓
predict()
   ↓
calculate score
```

You do **not** need to manually call `fit()` before `cross_val_score()`.

---

# Mean Cross-Validation Score

After cross-validation, you get multiple scores.

For example:

```text
[0.95, 0.95, 0.90, 0.95, 1.00]
```

The mean score is:

$$
\text{Mean} =
\frac{0.95+0.95+0.90+0.95+1.00}{5}
$$

```text
Mean = 0.95
```

Therefore:

```text
Mean CV Accuracy = 95%
```

The mean gives us the **average performance across all folds**.

---

# Standard Deviation

We can also calculate the standard deviation:

```python
print(scores.std())
```

For example:

```text
Mean = 0.95
Standard deviation = 0.0228
```

The standard deviation tells us how much the scores vary between folds.

A smaller standard deviation generally means the model's performance is **more consistent across the folds**.

```text
Low variation:

95%
94%
95%
96%
95%
```

```text
High variation:

70%
95%
80%
100%
75%
```

---

# Choosing the Number of Folds

You can change the number of folds:

```python
cv=3
```

```python
cv=5
```

```python
cv=10
```

For example:

```python
scores = cross_val_score(
    clf,
    X,
    y,
    cv=10
)
```

With 1,000 samples and 10 folds:

```text
1000 samples
     ↓
10 folds
     ↓
100 samples per validation fold
```

With more folds, each training set contains more data, but the model must be trained more times.

For example:

```text
5-fold  → 5 model fits
10-fold → 10 model fits
```

There is therefore a tradeoff between computational cost and how much data is used in each training/validation split.

---

# Cross-Validation vs Train/Test Split

A normal split might be:

```text
Dataset
   │
   ├── Training
   │
   └── Test
```

Cross-validation instead repeatedly creates training and validation portions:

```text
Dataset
   │
   ▼
K-Fold Cross-Validation
   │
   ├── Fold 1
   ├── Fold 2
   ├── Fold 3
   ├── Fold 4
   └── Fold 5
```

Cross-validation is useful when we want a more reliable estimate of model performance than a single split can provide.

---

# Important: Cross-Validation Does Not Return One Trained Model

When you run:

```python
scores = cross_val_score(clf, X, y, cv=5)
```

the models trained during the folds are used internally to calculate the scores.

`cross_val_score()` returns the **scores**, not a final trained model.

If you later want to use the model for predictions, you train it separately:

```python
clf.fit(X, y)

predictions = clf.predict(X_new)
```

---

# Important Concepts

### Fold

One portion of the dataset used during cross-validation.

### K-Fold

A cross-validation method that divides the dataset into `K` folds.

### Validation Score

The performance of the model on the fold that was held out during that iteration.

### Mean CV Score

The average score across all folds.

### Standard Deviation

Measures how much the validation scores vary between folds.

---

# Summary

Cross-validation repeatedly divides data into training and validation portions to estimate how well a machine learning model generalizes to unseen data.

The most common approach is **K-Fold Cross-Validation**.

The basic Scikit-learn API is:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)
```

Then:

```python
scores.mean()
```

gives the average cross-validation score.

And:

```python
scores.std()
```

shows the variation between the fold scores.

### Core idea

```text
Dataset
   ↓
Split into K folds
   ↓
Train + validate K times
   ↓
Get K scores
   ↓
Mean → average performance
Std  → variation between folds
```

**Cross-Validation = repeatedly train and validate on different portions of the data to get a more reliable estimate of model performance.**
