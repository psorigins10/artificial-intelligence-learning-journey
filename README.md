# 🤖 Artificial Intelligence Learning Journey

> My hands-on journey through Data Science, Machine Learning, and eventually Deep Learning.

This repository documents what I am learning, implementing, experimenting with, and understanding along the way.

The goal is not to simply learn how to use `scikit-learn`.

The goal is to understand **what the algorithms are doing, why they work, how they are evaluated, and where they fail**.

---

## 🧭 Learning Roadmap

```text
Data Science
     │
     ▼
Supervised Learning
     │
     ├── Regression
     │
     ├── Regularization
     │
     └── Classification
     │
     ▼
Unsupervised Learning
     │
     ▼
Deep Learning
     │
     ▼
Advanced AI
```

This repository is currently focused primarily on **Data Science and Supervised Learning**.

---

# 📚 What I Have Covered

## 📊 Data Science

### Pandas

- DataFrames
- Dataset exploration
- Searching/filtering datasets
- Working with structured data

📁 `04-data-science/pandas/`

### Matplotlib

- Basic plotting
- Data visualization
- Working with charts and figures

📁 `04-data-science/matplotlib/`

### Datasets

Currently used datasets include:

- Pokémon dataset
- House-price datasets
- Regression practice datasets

📁 `04-data-science/data-bases/`

---

# 🤖 Supervised Learning

## 📈 Regression

### 1. Linear Regression

Concepts covered:

- Linear Regression
- Features and targets
- Coefficients
- Intercept
- Predictions
- Best-fit line
- Ordinary Least Squares
- Multiple Linear Regression

📁 `01-supervised-learning/01-linear-regression/`

---

### 2. Train/Test Split

Concepts covered:

- Training data
- Testing data
- Generalization
- `train_test_split`
- `test_size`
- `random_state`
- Evaluating models on unseen data

📁 `01-supervised-learning/02-train-test-split/`

---

### 3. Regression Metrics

Metrics covered:

- Mean Absolute Error — MAE
- Mean Squared Error — MSE
- Root Mean Squared Error — RMSE
- R² Score

📁 `01-supervised-learning/03-regression-metrics/`

---

## 🧲 Regularization

### 4. Ridge Regression

Concepts covered:

- Overfitting
- L2 Regularization
- Regularization penalty
- Coefficient shrinkage
- `alpha`
- Ridge Regression with Scikit-learn
- Ridge Regression from scratch

📁 `01-supervised-learning/04-ridge-regression/`

---

### 5. Lasso Regression

Concepts covered:

- L1 Regularization
- Overfitting
- Coefficient shrinkage
- Feature selection
- Coefficients becoming zero
- Lasso Regression with Scikit-learn

📁 `01-supervised-learning/05-lasso-regression/`

---

### 6. Elastic Net

Concepts covered:

- L1 + L2 Regularization
- Combining Ridge and Lasso
- `alpha`
- `l1_ratio`
- Feature selection
- Handling correlated features

📁 `01-supervised-learning/06-elastic-net/`

---

# 🎯 Classification

### 7. Logistic Regression

Concepts covered:

- Classification vs Regression
- Binary Classification
- Linear combination
- `z = wx + b`
- Sigmoid function
- Probability prediction
- Classification threshold
- Logistic Regression with Scikit-learn

📁 `01-supervised-learning/07-logistic-regression/`

---

### 8. Classification Metrics

Metrics and concepts covered:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1 Score
- Log Loss
- Model Complexity

📁 `01-supervised-learning/08-classification-metrics/`

---

### 9. Decision Trees

Concepts covered:

- Decision Tree intuition
- Decision rules
- Splitting data
- Classification
- Feature-based decisions
- Model interpretation
- Scikit-learn Decision Trees
- Pokémon classification experiment

📁 `01-supervised-learning/09-decision-tree/`

---

# 🗂️ Repository Structure

```text
artificial-intelligence-learning-journey/
│
├── 01-supervised-learning/
│   │
│   ├── 01-linear-regression/
│   │   ├── 01-polynomial-regression/
│   │   ├── housePricePrediction.py
│   │   ├── linearRegressionBasic.py
│   │   └── readme.md
│   │
│   ├── 02-train-test-split/
│   │   ├── train_test_split.py
│   │   └── readme.md
│   │
│   ├── 03-regression-metrics/
│   │   ├── 01-mae/
│   │   ├── 02-mse/
│   │   ├── 03-rmse/
│   │   └── 04-r2-score/
│   │
│   ├── 04-ridge-regression/
│   │   ├── 01-ridge-regression.py
│   │   ├── 02-ridge_from_scratch.py
│   │   └── readme.md
│   │
│   ├── 05-lasso-regression/
│   │   ├── 01-lasso-regression.py
│   │   └── README.md
│   │
│   ├── 06-elastic-net/
│   │   ├── 01-elastic-net.py
│   │   └── README.md
│   │
│   ├── 07-logistic-regression/
│   │   ├── 01-logistic-regression.py
│   │   └── README.md
│   │
│   ├── 08-classification-metrics/
│   │   ├── 01-accuracy-score/
│   │   ├── 02-confusion-matrix/
│   │   ├── 03-f1-score/
│   │   ├── 04-model-complexity/
│   │   └── 05-log-loss/
│   │
│   ├── 09-decision-tree/
│   │    ├── images/
│   │    ├── 01-decision-tree.py
│   │    └── README.md
│   │
│   ├── 10-random-forest/
│   │     ├── 01-random-forest.py
│   │     └── README.md
│   │
│   ├── 11-knn/
│   │    ├── images/
│   │    ├── 01-knn.py
│   │    └── README.md
│   │
│   └── 12-naive-bayes/
│        ├── images/
│        ├── 01-naive-bayes.py
│        └── README.md
│
├── 04-data-science/
│   │
│   ├── data-bases/
│   │   ├── house_price_lasso_practice.csv
│   │   ├── house_price_ridge_practice.csv
│   │   └── pokemon.csv
│   │
│   ├── matplotlib/
│   │   ├── matplotlib_learn1.py
│   │   └── matplotlib_learn2.py
│   │
│   └── pandas/
│       └── search_pokemon001.py
│
├── .gitignore
└── README.md
```

---

# 🧠 My Learning Process

For each topic, I try to follow this cycle:

```text
        Learn the Concept
               │
               ▼
     Understand the Mathematics
               │
               ▼
       Understand the Algorithm
               │
               ▼
          Write the Code
               │
               ▼
       Experiment with Data
               │
               ▼
         Evaluate the Model
               │
               ▼
        Document What I Learned
```

The objective is to move beyond:

```python
model.fit(X, y)
model.predict(X)
```

and understand what happens **inside the model**.

---

# 🛠️ Tools & Technologies

| Technology | Purpose |
|---|---|
| Python | Programming |
| NumPy | Numerical computing |
| Pandas | Data manipulation |
| Matplotlib | Data visualization |
| Scikit-learn | Machine Learning |
| Git | Version control |
| GitHub | Code and documentation |

---

# 📈 Progress

## ✅ Completed

- [x] Data Science fundamentals
- [x] Pandas
- [x] Matplotlib
- [x] Linear Regression
- [x] Polynomial Regression
- [x] Train/Test Split
- [x] MAE
- [x] MSE
- [x] RMSE
- [x] R² Score
- [x] Ridge Regression
- [x] L2 Regularization
- [x] Overfitting
- [x] Lasso Regression
- [x] L1 Regularization
- [x] Feature Selection
- [x] Elastic Net
- [x] Logistic Regression
- [x] Sigmoid Function
- [x] Binary Classification
- [x] Accuracy
- [x] Confusion Matrix
- [x] Precision
- [x] Recall
- [x] F1 Score
- [x] Log Loss
- [x] Model Complexity
- [x] Decision Trees
- [x] Random Forest
- [x] K-Nearest Neighbors
- [x] Naive Bayes
- [x] Support Vector Machines

## 🔨 Coming Next

These are planned learning topics, not topics I am claiming to have completed:

- [ ] Cross-Validation
- [ ] Hyperparameter Tuning
- [ ] Ensemble Learning
- [ ] Unsupervised Learning
- [ ] K-Means Clustering
- [ ] DBSCAN
- [ ] PCA
- [ ] Deep Learning

---

# 🎯 Why I'm Building This

A lot of Machine Learning tutorials teach you to use an API without really understanding the model.

I want to approach it differently.

For every algorithm, I want to understand:

- What problem does it solve?
- What assumptions does it make?
- What is happening mathematically?
- How does the algorithm learn?
- What parameters control it?
- How can it overfit?
- How do we evaluate it?
- When should we use it?
- When should we **not** use it?

This repository is my attempt to answer those questions through:

**Code + Mathematics + Experiments + Documentation**

---

# 🚧 Status

This repository is actively being developed.

I'm learning one concept at a time, implementing it, experimenting with datasets, and documenting what I understand.

The structure will change as the journey moves from:

```text
Data Science
      ↓
Supervised Learning
      ↓
Unsupervised Learning
      ↓
Deep Learning
      ↓
Advanced AI
```

---

## 🤝 Let's Build Shit Together

This isn't a finished textbook.

It's a record of learning AI from the fundamentals.

If something is wrong, inefficient, or poorly implemented, that's part of the process.

**Keep learning. Keep building. Keep breaking things.**

---

⭐ **Repository:**  
https://github.com/psorigins10/artificial-intelligence-learning-journey
