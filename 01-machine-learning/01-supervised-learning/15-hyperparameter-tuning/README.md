# Hyperparameter Tuning

## What is Hyperparameter Tuning?

**Hyperparameter tuning** is the process of finding the best values for a machine learning model's **hyperparameters**.

Hyperparameters are settings that we choose **before training the model**.

For example, an SVM has hyperparameters such as:

```python
SVC(
    C=1,
    kernel="rbf",
    gamma="scale"
)
```

Here:

```text
C       → Hyperparameter
kernel  → Hyperparameter
gamma   → Hyperparameter
```

The model does not learn these values automatically during `fit()`.

We choose them, or use a search method to find good values.

---

# Parameters vs Hyperparameters

This distinction is important.

## Parameters

Parameters are values that the model **learns from the training data**.

For example, in a linear model:

$$
w_1, w_2, ..., w_n, b
$$

are learned during training.

```python
model.fit(X, y)
```

The model calculates these values from the data.

---

## Hyperparameters

Hyperparameters are values that control **how the model behaves**.

For example:

```python
SVC(
    C=10,
    kernel="rbf"
)
```

The values `C` and `kernel` are selected before training.

The model does not learn them in the same way it learns its parameters.

---

# Why Do We Need Hyperparameter Tuning?

Suppose we train an SVM:

```python
clf = SVC(C=1)
```

We don't know whether `C=1` is the best choice.

Maybe:

```text
C = 0.1 → 91%
C = 1   → 95%
C = 10  → 97%
C = 100 → 93%
```

If `C=10` performs best, we would prefer:

```python
SVC(C=10)
```

Instead of guessing the value manually, we can systematically test different values.

This process is called **hyperparameter tuning**.

---

# What is Grid Search?

**Grid Search** is a method that systematically tries different combinations of hyperparameter values.

In Scikit-learn, this is implemented using:

```python
GridSearchCV
```

Import:

```python
from sklearn.model_selection import GridSearchCV
```

---

# Basic GridSearchCV Example

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    random_state=42
)

clf = SVC()

param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}

grid = GridSearchCV(
    clf,
    param_grid,
    cv=5
)

grid.fit(X, y)

print(grid.best_params_)
print(grid.best_score_)
```

---

# Understanding `param_grid`

The `param_grid` contains the hyperparameters and the values we want to test.

```python
param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}
```

There are three hyperparameters:

```text
C
kernel
gamma
```

with:

```text
C       → 3 values
kernel  → 2 values
gamma   → 2 values
```

Grid Search tests every combination.

Therefore:

$$
3 \times 2 \times 2 = 12
$$

different combinations.

---

# Hyperparameter Combinations

GridSearchCV conceptually tests:

```text
C=0.1, kernel=linear, gamma=scale
C=0.1, kernel=linear, gamma=auto

C=0.1, kernel=rbf,    gamma=scale
C=0.1, kernel=rbf,    gamma=auto

C=1,   kernel=linear, gamma=scale
C=1,   kernel=linear, gamma=auto

C=1,   kernel=rbf,    gamma=scale
C=1,   kernel=rbf,    gamma=auto

C=10,  kernel=linear, gamma=scale
C=10,  kernel=linear, gamma=auto

C=10,  kernel=rbf,    gamma=scale
C=10,  kernel=rbf,    gamma=auto
```

It then compares the performance of these configurations.

---

# GridSearchCV and Cross-Validation

`GridSearchCV` uses **cross-validation** to evaluate each hyperparameter combination.

For example:

```python
grid = GridSearchCV(
    clf,
    param_grid,
    cv=5
)
```

If there are:

```text
12 hyperparameter combinations
```

and:

```text
5 CV folds
```

then GridSearchCV performs:

$$
12 \times 5 = 60
$$

model fits.

Conceptually:

```text
Hyperparameter combination 1
        ↓
     5-fold CV
        ↓
     CV score

Hyperparameter combination 2
        ↓
     5-fold CV
        ↓
     CV score

          ...

Hyperparameter combination 12
        ↓
     5-fold CV
        ↓
     CV score
```

The best-performing combination is selected.

---

# `cv` vs Hyperparameters

A common beginner mistake is treating `cv` as a model hyperparameter.

For example, this is incorrect:

```python
param_grid = {
    "C": [0.1, 1, 10],
    "cv": [3, 5, 10]
}
```

`cv` controls **how GridSearchCV evaluates the model**.

It is not an SVM hyperparameter.

Instead:

```python
grid = GridSearchCV(
    clf,
    param_grid,
    cv=5
)
```

Think of it as:

```text
C
→ How should the SVM behave?

kernel
→ What type of decision function should it use?

gamma
→ How should the kernel behave?

cv
→ How should GridSearchCV evaluate each configuration?
```

---

# `best_params_`

After fitting:

```python
grid.fit(X, y)
```

you can access:

```python
grid.best_params_
```

For example:

```text
{'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}
```

This means that, among the combinations you provided, this configuration achieved the best CV performance.

---

# `best_score_`

You can also get:

```python
grid.best_score_
```

For example:

```text
0.98
```

This means the best configuration achieved a mean cross-validation score of:

```text
98%
```

---

# `best_estimator_`

GridSearchCV also provides the best estimator:

```python
grid.best_estimator_
```

For example:

```python
best_model = grid.best_estimator_
```

You can then use it for predictions:

```python
predictions = best_model.predict(X_new)
```

The estimator contains the best hyperparameter configuration found by the search.

---

# How Many Hyperparameters Can We Tune?

There is no fixed number.

You can provide multiple hyperparameters:

```python
param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}
```

The important limitation is the **number of combinations**.

For example:

```text
4 C values
×
3 kernel values
×
5 gamma values
×
2 degree values
```

gives:

$$
4 \times 3 \times 5 \times 2 = 120
$$

combinations.

With 5-fold CV:

$$
120 \times 5 = 600
$$

model fits.

So a large grid can become computationally expensive very quickly.

---

# GridSearchCV Workflow

The complete process is:

```text
              Dataset
                 ↓
          Candidate Model
                 ↓
          Define Parameter Grid
                 ↓
        ┌────────────────────┐
        │   GridSearchCV     │
        └────────────────────┘
                 ↓
     Try every combination
                 ↓
        Cross-validation
                 ↓
        Compare CV scores
                 ↓
         Best parameters
                 ↓
          Best estimator
```

---

# Example with Multiple Models

Hyperparameter tuning can be performed on many different machine learning models.

For example, an SVM:

```python
param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"]
}
```

A Random Forest:

```python
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10]
}
```

A KNN model:

```python
param_grid = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"]
}
```

The available hyperparameters depend on the model.

---

# GridSearchCV vs Manual Tuning

Without GridSearchCV:

```python
SVC(C=0.1)
SVC(C=1)
SVC(C=10)
SVC(C=100)
```

You would have to manually train and evaluate each model.

With GridSearchCV:

```python
param_grid = {
    "C": [0.1, 1, 10, 100]
}

grid = GridSearchCV(
    SVC(),
    param_grid,
    cv=5
)

grid.fit(X, y)
```

GridSearchCV handles the search and evaluation automatically.

---

# Grid Search Limitations

Grid Search has an important limitation:

**It tries every combination.**

Suppose you have:

```text
10 values × 10 values × 10 values
```

That's:

$$
10^3 = 1000
$$

combinations.

With 5-fold CV:

$$
1000 \times 5 = 5000
$$

model fits.

This can become very expensive.

For large search spaces, another technique called **Randomized Search** can be more efficient.

Scikit-learn provides:

```python
from sklearn.model_selection import RandomizedSearchCV
```

Randomized Search samples combinations instead of testing every possible combination.

---

# Important: Hyperparameter Tuning Does Not Guarantee the Best Model

GridSearchCV finds the best configuration **from the values you provide**.

For example:

```python
param_grid = {
    "C": [0.1, 1, 10]
}
```

If the true useful value is `C=50`, GridSearchCV cannot find it because you never gave it `50` as an option.

Therefore:

```text
Grid Search
≠
Find the universally perfect hyperparameters
```

Instead:

```text
Grid Search
=
Find the best configuration among the configurations you searched
```

---

# Key Terms

### Hyperparameter

A model setting chosen before training.

Examples:

```text
C
kernel
gamma
n_estimators
max_depth
n_neighbors
```

### Parameter

A value learned by the model during training.

### Parameter Grid

A dictionary containing the hyperparameters and values to search.

```python
param_grid = {
    "C": [0.1, 1, 10]
}
```

### Grid Search

Systematically tests every combination in the parameter grid.

### `GridSearchCV`

Scikit-learn's implementation of grid search using cross-validation.

### `best_params_`

Returns the best hyperparameter combination found.

### `best_score_`

Returns the best mean cross-validation score.

### `best_estimator_`

Returns the estimator configured with the best hyperparameters.

---

# Summary

**Hyperparameter tuning** is the process of searching for good hyperparameter values for a machine learning model.

**GridSearchCV** automates this process by:

```text
1. Taking a model
2. Taking a parameter grid
3. Creating every possible combination
4. Evaluating each combination using cross-validation
5. Comparing the results
6. Selecting the best combination
```

The most important API is:

```python
from sklearn.model_selection import GridSearchCV
```

Basic usage:

```python
grid = GridSearchCV(
    model,
    param_grid,
    cv=5
)

grid.fit(X, y)
```

Then:

```python
grid.best_params_
```

returns the best hyperparameters,

```python
grid.best_score_
```

returns the best mean CV score,

and:

```python
grid.best_estimator_
```

returns the best fitted model.

## Core Idea

```text
Hyperparameters
      ↓
Parameter Grid
      ↓
GridSearchCV
      ↓
Try combinations
      ↓
Cross-validation
      ↓
Compare scores
      ↓
Best hyperparameters
      ↓
Best estimator
```

> **Hyperparameter tuning = searching for the best model settings.**

> **GridSearchCV = a tool that systematically searches those settings and evaluates them using cross-validation.**
