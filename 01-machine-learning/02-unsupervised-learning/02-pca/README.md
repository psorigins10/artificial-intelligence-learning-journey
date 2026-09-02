# Principal Component Analysis (PCA)

## What is PCA?

**PCA (Principal Component Analysis)** is an unsupervised learning algorithm used for **dimensionality reduction**.

Its main purpose is to transform a dataset with many features into a smaller number of new features called **principal components**, while preserving as much of the important variation in the original data as possible.

For example:

```text
Original dataset

Feature 1
Feature 2
Feature 3
Feature 4
Feature 5
Feature 6

        ↓ PCA

Principal Component 1
Principal Component 2
```

This reduces:

```text
6 dimensions → 2 dimensions
```

PCA is especially useful for:

- Visualizing high-dimensional data
- Reducing the number of features
- Removing redundant information
- Speeding up some machine learning models
- Finding the most important directions of variation in data

---

# 1. Why Do We Need PCA?

Imagine a dataset containing six Pokémon statistics:

```text
HP
Attack
Defense
Sp. Attack
Sp. Defense
Speed
```

Each Pokémon is a point in **6-dimensional space**.

Humans cannot directly visualize six dimensions.

PCA can transform those six dimensions into two dimensions:

```text
6D data
  ↓
 PCA
  ↓
2D data
```

We can then plot the two principal components.

---

# 2. The Main Idea Behind PCA

PCA looks for new directions (axes) in the data.

The first direction captures the **maximum variance** in the data.

This direction is called:

```text
PC1
```

The second direction captures the maximum remaining variance while being orthogonal to PC1.

This is:

```text
PC2
```

Then:

```text
PC3
PC4
...
```

can capture additional variance.

The components are ordered by how much variance they explain:

```text
PC1 → most variance
PC2 → second most
PC3 → third most
...
```

---

# 3. What Does "Variance" Mean?

Variance describes how spread out the data is.

Suppose the values of a feature are:

```text
10, 10, 10, 10, 10
```

There is almost no variation.

But:

```text
1, 20, 50, 80, 100
```

has much more variation.

PCA tries to find directions where the data has the greatest variation.

The intuition is:

> A direction with greater variance often contains more information about the structure of the dataset.

---

# 4. PCA Does NOT Simply Select Features

This is an important point.

Suppose we start with:

```text
X1 = HP
X2 = Attack
X3 = Defense
X4 = Sp. Attack
X5 = Sp. Defense
X6 = Speed
```

PCA does **not** simply say:

```text
Keep Attack and Speed
```

Instead, it creates new features by combining the original features.

Conceptually:

\[
PC_1 =
w_1X_1+w_2X_2+w_3X_3+\cdots+w_6X_6
\]

and:

\[
PC_2 =
v_1X_1+v_2X_2+v_3X_3+\cdots+v_6X_6
\]

The weights determine how strongly each original feature contributes to each principal component.

---

# 5. Principal Components

A principal component is a **new axis / feature created from the original features**.

For example, PCA might produce:

```text
Original features:

HP
Attack
Defense
Sp. Attack
Sp. Defense
Speed

        ↓

New features:

PC1
PC2
PC3
PC4
PC5
PC6
```

The number of principal components can be at most the number of original features.

If we only need two dimensions:

```python
PCA(n_components=2)
```

then PCA produces:

```text
PC1
PC2
```

---

# 6. PCA Workflow

A simplified PCA workflow is:

```text
Original Data
     ↓
Select Features
     ↓
Scale Features
     ↓
Center Data
     ↓
Calculate Directions of Maximum Variance
     ↓
Create Principal Components
     ↓
Choose Number of Components
     ↓
Transform Data
```

---

# 7. Feature Scaling

Scaling is usually important before applying PCA.

Suppose:

```text
Feature A: 0–100
Feature B: 0–1,000,000
```

The second feature has a much larger numerical scale.

Without scaling, it can have a disproportionately large influence on PCA.

A common approach is standardization:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Standardization transforms each feature approximately so that:

```text
mean = 0
standard deviation = 1
```

Then PCA can be applied:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)
```

### Important

Scaling is not always mandatory. If all features are already on comparable scales and their original units are meaningful, you may choose not to standardize them.

---

# 8. PCA with Scikit-Learn

The basic API is:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)
```

Here:

```python
n_components=2
```

means:

> Transform the data into two principal components.

If:

```python
X.shape
```

is:

```text
(800, 6)
```

then:

```python
X_pca.shape
```

will be:

```text
(800, 2)
```

So:

```text
800 observations × 6 features
             ↓
            PCA
             ↓
800 observations × 2 components
```

---

# 9. Visualizing PCA

After reducing the data to two dimensions:

```python
import matplotlib.pyplot as plt

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1]
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title("PCA")

plt.show()
```

The axes are:

```text
X-axis → PC1
Y-axis → PC2
```

Each point represents one observation from the original dataset.

---

# 10. Explained Variance

One of the most important concepts in PCA is **explained variance**.

Every principal component explains some amount of the total variance in the original dataset.

Scikit-learn provides this through:

```python
pca.explained_variance_ratio_
```

For example:

```python
print(pca.explained_variance_ratio_)
```

might return:

```text
[0.45, 0.25]
```

This means:

```text
PC1 → 45%
PC2 → 25%
```

Together:

```text
45% + 25% = 70%
```

So the two components preserve approximately **70% of the variance** in the original data.

You can calculate the total:

```python
print(pca.explained_variance_ratio_.sum())
```

---

# 11. Choosing the Number of Components

You don't always have to manually choose:

```python
n_components=2
```

You can ask PCA to retain a certain percentage of the variance.

For example:

```python
pca = PCA(n_components=0.95)

X_pca = pca.fit_transform(X_scaled)
```

This tells PCA:

> Keep enough components to explain approximately 95% of the variance.

The resulting number of components can then be inspected:

```python
print(pca.n_components_)
```

For example:

```text
Original features: 50
Components selected: 12
```

This means PCA reduced:

```text
50 features → 12 components
```

while retaining approximately 95% of the variance.

---

# 12. Cumulative Explained Variance

You can examine how much variance is retained as more components are added:

```python
import numpy as np

pca = PCA()

pca.fit(X_scaled)

cumulative_variance = np.cumsum(
    pca.explained_variance_ratio_
)

print(cumulative_variance)
```

You can visualize it:

```python
import matplotlib.pyplot as plt

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker="o"
)

plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")

plt.show()
```

This helps answer:

> "How many principal components do I need?"

---

# 13. PCA Components and Feature Contributions

Scikit-learn stores the principal component directions in:

```python
pca.components_
```

For example:

```python
print(pca.components_)
```

If there are six original features and two components:

```text
2 × 6
```

The rows correspond to:

```text
PC1
PC2
```

and the columns correspond to the original features.

For example, conceptually:

```text
          HP   Attack   Defense   SpAtk   SpDef   Speed

PC1      0.4    0.5      0.3      0.4     0.3     0.2

PC2     -0.1    0.2      0.6     -0.3     0.5    -0.4
```

The magnitude of the values indicates how strongly the original features contribute to the component.

---

# 14. PCA and Reconstruction

PCA can also approximately reconstruct the original data from the reduced representation.

For example:

```python
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

X_reconstructed = pca.inverse_transform(X_pca)
```

The reconstructed data will generally not be identical to the original data because information was discarded during dimensionality reduction.

For example:

```text
Original:
6 dimensions

        ↓ PCA

2 dimensions

        ↓ inverse_transform

Approximate:
6 dimensions
```

The fewer components you keep, the more information may be lost.

---

# 15. PCA for Visualization

One of the most common uses of PCA is visualizing high-dimensional datasets.

Suppose:

```text
100 features
```

You cannot easily make a normal 100-dimensional plot.

PCA can reduce the data:

```text
100D
 ↓
PCA
 ↓
2D
```

Then:

```python
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1]
)
```

This gives a 2D representation of the original high-dimensional data.

### Important

A PCA plot is a **projection/representation** of the original data.

It does not mean the original data actually only has two dimensions.

---

# 16. PCA with K-Means

PCA and K-Means solve different problems.

### K-Means

```text
Find groups
```

### PCA

```text
Reduce dimensions
```

They can be combined.

For example:

```text
Original 6D data
       ↓
    K-Means
       ↓
Cluster labels
       ↓
      PCA
       ↓
     2D plot
```

The PCA transformation lets you visualize the clustering.

Example:

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init="auto"
)

labels = kmeans.fit_predict(X_scaled)

# PCA
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

# Visualization
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=labels
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
```

Here K-Means still operates on the six-dimensional scaled data.

PCA is only being used to create a two-dimensional representation for visualization.

---

# 17. Important PCA Limitations

PCA is useful, but it is not magic.

### Information can be lost

Reducing:

```text
100 dimensions → 2 dimensions
```

will usually discard some information.

### Principal components may be hard to interpret

A component can be a combination of many original features.

Instead of:

```text
PC1 = Attack
```

you may get:

```text
PC1 =
0.45 × HP
+ 0.51 × Attack
+ 0.31 × Defense
+ ...
```

So PC1 may not have an obvious real-world meaning.

### PCA is sensitive to scaling

Features with large numerical scales can dominate the result.

### PCA is linear

Standard PCA looks for linear combinations of the original features. It may not capture complicated nonlinear structures well.

---

# 18. Important PCA Terminology

| Term | Meaning |
|---|---|
| Feature | Original variable in the dataset |
| Principal Component | New variable created by PCA |
| PC1 | Component explaining the most variance |
| PC2 | Component explaining the next most variance |
| Variance | Measure of data spread |
| Explained Variance Ratio | Percentage of variance explained by a component |
| Components | New axes created by PCA |
| `n_components` | Number of components to keep |
| `components_` | Principal component directions in scikit-learn |

---

# 19. Complete PCA Example

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("pokemon.csv")

features = [
    "HP",
    "Attack",
    "Defense",
    "Sp. Atk",
    "Sp. Def",
    "Speed"
]

X = df[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

# Explained variance
print("Explained variance:")
print(pca.explained_variance_ratio_)

print("Total explained variance:")
print(pca.explained_variance_ratio_.sum())

# Plot
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1]
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Pokémon Stats")

plt.show()
```

---

# 20. PCA Mental Model

The easiest way to remember PCA is:

```text
Many Features
      ↓
Find the directions where
the data varies the most
      ↓
Create new axes
      ↓
PC1, PC2, PC3, ...
      ↓
Keep the most useful components
      ↓
Fewer dimensions
```

For example:

```text
6D
 ↓
PCA
 ↓
PC1 + PC2
 ↓
2D visualization
```

## Final Takeaway

**PCA is a dimensionality-reduction algorithm that transforms the original features into new principal components ordered by the amount of variance they explain.**

The core idea is:

\[
\boxed{\text{Find directions of maximum variance and represent the data using fewer of those directions}}
\]

PCA does not directly predict a target and does not create clusters. Its job is to **transform and reduce the representation of the data**.
