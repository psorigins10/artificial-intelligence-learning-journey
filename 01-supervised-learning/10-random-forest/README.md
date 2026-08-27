# 🌲 Random Forest

Random Forest is an **ensemble learning algorithm** that combines multiple Decision Trees to make a stronger and more stable prediction.

Since Decision Trees are already covered separately in this repository, this README focuses on:

- What Random Forest actually is
- What happens when the model trains
- Why multiple trees are used
- Bootstrap sampling
- Random feature selection
- How predictions are combined
- Why Random Forest usually performs better than a single tree
- Important Random Forest parameters
- Feature importance

---

# 🧠 The Core Idea

A single Decision Tree can become very sensitive to the training data.

If we slightly change the training data, the tree can change significantly.

Random Forest solves this by creating **many different Decision Trees** and combining their predictions.

```text
                  RANDOM FOREST
                        │
          ┌─────────────┼─────────────┐
          │             │             │
          ▼             ▼             ▼
       Tree 1        Tree 2        Tree 3       ... Tree N
          │             │             │
          ▼             ▼             ▼
      Prediction    Prediction    Prediction
          │             │             │
          └─────────────┼─────────────┘
                        │
                        ▼
                 Combine Results
                        │
                        ▼
                  Final Prediction
```

Instead of trusting one tree, Random Forest asks **many trees** and combines their answers.

---

# 🌳 What Is Actually Happening?

Suppose we have a Pokémon dataset and want to predict:

```text
Is this Pokémon Legendary?
```

Our features are:

```text
HP
Attack
Defense
Sp. Atk
Sp. Def
Speed
Generation
```

And our target is:

```text
Legendary
```

When we create:

```python
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

we are asking Scikit-learn to build:

```text
100 Decision Trees
```

But the important question is:

> Why don't all 100 trees become identical?

That's where the randomness comes in.

---

# 🎲 Randomness #1 — Bootstrap Sampling

Random Forest creates different training datasets for different trees using **bootstrap sampling**.

Imagine our original training dataset contains:

```text
800 samples
```

Random Forest creates a sample for Tree 1:

```text
Tree 1
├── Pokémon 12
├── Pokémon 421
├── Pokémon 12
├── Pokémon 87
├── Pokémon 654
├── Pokémon 231
└── ...
```

Tree 2 gets another sample:

```text
Tree 2
├── Pokémon 91
├── Pokémon 421
├── Pokémon 302
├── Pokémon 91
├── Pokémon 17
├── Pokémon 654
└── ...
```

Tree 3 gets another:

```text
Tree 3
├── Pokémon 52
├── Pokémon 17
├── Pokémon 421
├── Pokémon 733
├── Pokémon 52
├── Pokémon 188
└── ...
```

Notice that some samples can appear **multiple times**.

Some samples may not appear in a particular tree's training sample at all.

This is called **bootstrap sampling**.

---

# 🎲 Randomness #2 — Random Feature Selection

Random Forest doesn't only randomize the training samples.

It also introduces randomness in the features considered by each tree/split.

Instead of always allowing every feature to compete at every split, Random Forest considers a **random subset of features**.

For example:

```text
All Features
─────────────────────────────
HP
Attack
Defense
Sp. Atk
Sp. Def
Speed
Generation
```

One split might consider:

```text
Attack
Defense
Speed
```

Another might consider:

```text
HP
Sp. Def
Generation
```

Another:

```text
Attack
Sp. Atk
Speed
```

This forces the trees to become less correlated with each other.

---

# 🌲 So What Does One Tree Do?

Each tree trains on:

```text
Different bootstrap sample
        +
Random subsets of features
        ↓
     Tree grows
        ↓
     Prediction
```

For example:

```text
Tree 1 → Legendary
Tree 2 → Not Legendary
Tree 3 → Legendary
Tree 4 → Legendary
Tree 5 → Not Legendary
...
```

The trees don't have to agree.

That's actually the point.

---

# 🗳️ How Classification Works

For classification, Random Forest uses the predictions from all the trees and combines them through voting.

Suppose we have 5 trees:

```text
Tree 1 → Legendary
Tree 2 → Legendary
Tree 3 → Not Legendary
Tree 4 → Legendary
Tree 5 → Not Legendary
```

Votes:

```text
Legendary     = 3
Not Legendary = 2
```

Therefore:

```text
Final Prediction = Legendary
```

With 100 trees:

```text
Legendary     = 73
Not Legendary = 27
```

Final prediction:

```text
Legendary
```

This is essentially **majority voting** for classification.

---

# 🔬 What Happens During `fit()`?

When we write:

```python
model.fit(X_train, y_train)
```

Random Forest performs roughly this process:

```text
              Training Dataset
                     │
                     ▼
           Create bootstrap sample
                     │
                     ▼
               Build Tree 1
                     │
                     ▼
           Create another sample
                     │
                     ▼
               Build Tree 2
                     │
                     ▼
                     ...
                     │
                     ▼
              Build Tree 100
                     │
                     ▼
              Store all trees
```

Each tree is trained independently using its own randomized data/features.

The final model is therefore not one giant tree.

It is a **collection of trained trees**.

---

# 🔮 What Happens During `predict()`?

When we write:

```python
y_pred = model.predict(X_test)
```

the process is approximately:

```text
                  New Pokémon
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Tree 1       Tree 2       Tree 3     ... Tree 100
          │            │            │
          ▼            ▼            ▼
     Legendary    Not Legendary  Legendary
          │            │            │
          └────────────┼────────────┘
                       ▼
                 Count the votes
                       │
                       ▼
               Majority decision
                       │
                       ▼
                Final Prediction
```

---

# 🧩 Why Does This Help?

The biggest advantage comes from **reducing variance**.

A single Decision Tree can be unstable:

```text
Small change in data
        ↓
Different tree
        ↓
Different prediction
```

Random Forest averages/combine many different trees:

```text
Tree 1 ─┐
Tree 2 ─┤
Tree 3 ─┤
Tree 4 ─┤──→ Combined prediction
Tree 5 ─┤
  ...   │
Tree N ─┘
```

Individual trees can make mistakes.

But if their mistakes are not perfectly correlated, combining them can produce a more reliable model.

---

# ⚖️ Single Tree vs Random Forest

| Decision Tree | Random Forest |
|---|---|
| One tree | Many trees |
| Can have high variance | Usually lower variance |
| Easy to visualize | Harder to visualize as a whole |
| Can overfit easily | Generally more robust |
| Fast to train | More computationally expensive |
| Highly interpretable | Less interpretable |
| One model | Ensemble of models |

Random Forest sacrifices some interpretability in exchange for better stability and usually stronger predictive performance.

---

# 🔢 `n_estimators`

One of the most important parameters is:

```python
RandomForestClassifier(
    n_estimators=100
)
```

`n_estimators` controls the number of trees.

For example:

```text
n_estimators = 10
       ↓
10 trees

n_estimators = 100
       ↓
100 trees

n_estimators = 500
       ↓
500 trees
```

More trees generally make the forest more stable, but they also increase computation time and memory usage.

More trees do **not** automatically mean a better model forever.

---

# 🎛️ Important Parameters

## `n_estimators`

Number of trees in the forest.

```python
RandomForestClassifier(n_estimators=100)
```

---

## `max_depth`

Controls the maximum depth of each tree.

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10
)
```

Limiting tree depth can help control model complexity.

---

## `min_samples_split`

Minimum number of samples required to split an internal node.

```python
RandomForestClassifier(
    min_samples_split=5
)
```

---

## `min_samples_leaf`

Minimum number of samples that must exist in a leaf.

```python
RandomForestClassifier(
    min_samples_leaf=2
)
```

---

## `max_features`

Controls how many features are considered when looking for a split.

```python
RandomForestClassifier(
    max_features="sqrt"
)
```

This is one of the mechanisms that makes the individual trees different.

---

# 🧪 Our Pokémon Example

The model in this folder uses:

```python
from sklearn.ensemble import RandomForestClassifier
```

Features:

```python
X = df[
    [
        "HP",
        "Attack",
        "Defense",
        "Sp. Atk",
        "Sp. Def",
        "Speed",
        "Generation"
    ]
]
```

Target:

```python
y = df["Legendary"]
```

The model:

```python
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

This creates a forest containing:

```text
100 randomized Decision Trees
```

Each tree learns from randomized training data and randomized feature subsets.

The final prediction is produced by combining the predictions of those trees.

---

# 🔍 Looking Inside the Forest

Scikit-learn stores the individual trees inside:

```python
model.estimators_
```

For example:

```python
first_tree = model.estimators_[0]
```

You can check how many trees exist:

```python
print(len(model.estimators_))
```

Output:

```text
100
```

You can also visualize one tree:

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))

plot_tree(
    model.estimators_[0],
    feature_names=X.columns,
    class_names=["Not Legendary", "Legendary"],
    filled=True,
    max_depth=3
)

plt.show()
```

We only visualize one tree because visualizing all 100 at once would be mostly useless.

---

# 📊 Feature Importance

Random Forest can also estimate how useful each feature was for making predictions.

```python
print(model.feature_importances_)
```

A better visualization:

```python
import pandas as pd
import matplotlib.pyplot as plt

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values()

importance.plot(kind="barh")

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

plt.show()
```

This helps answer:

> Which Pokémon attributes were most useful for predicting whether a Pokémon is Legendary?

---

# 🧠 The Big Picture

Random Forest can be understood as:

```text
                  DATASET
                     │
                     ▼
          ┌─────────────────────┐
          │ Bootstrap Sampling  │
          └──────────┬──────────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Sample 1   Sample 2   Sample 3
          │          │          │
          ▼          ▼          ▼
        Tree 1     Tree 2     Tree 3
          │          │          │
          └──────────┼──────────┘
                     │
              ... More Trees ...
                     │
                     ▼
              Tree Predictions
                     │
                     ▼
                 Voting
                     │
                     ▼
             Final Prediction
```

The important idea is:

> **Random Forest does not make one Decision Tree better. It makes many different trees and combines them.**

---

# ⚠️ What Random Forest Does NOT Mean

Random Forest does **not** mean:

```text
One huge Decision Tree
```

It means:

```text
Many independent-ish Decision Trees
              +
Randomized training samples
              +
Randomized feature selection
              ↓
        Combined prediction
```

The power comes from the **ensemble**, not from making one tree extremely complicated.

---

# 🚀 Key Takeaways

### 1. Random Forest is an ensemble

It combines many Decision Trees.

### 2. Randomness is intentional

Randomness comes mainly from:

- Bootstrap samples
- Random feature selection

### 3. Trees don't need to agree

Their diversity is useful.

### 4. Classification uses voting

The class receiving the most votes becomes the prediction.

### 5. More trees generally improve stability

But they also increase computational cost.

### 6. Random Forest reduces variance

Combining multiple diverse trees makes the model less dependent on the quirks of one particular training sample.

### 7. It is harder to interpret than one tree

You gain predictive robustness but lose some of the simple interpretability of a single Decision Tree.

---

# 📌 One Sentence Summary

> **Random Forest builds many diverse Decision Trees using randomized data and feature selection, then combines their predictions to produce a more stable and robust model.**
