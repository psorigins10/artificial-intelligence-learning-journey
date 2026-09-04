# 🤖 Artificial Intelligence Learning Journey

> My hands-on journey through Data Science, Machine Learning, Deep Learning, and eventually Advanced AI.

This repository documents what I am learning, implementing, experimenting with, breaking, fixing, and understanding along the way.

The goal is **not** to simply learn how to call machine-learning APIs.

The goal is to understand:

* What the algorithms are doing
* Why they work
* What is happening mathematically
* How models learn
* How models are evaluated
* Why models fail
* When to use a particular algorithm
* When **not** to use it

This is a learning repository, not a collection of polished production projects.

---

# 🧭 Learning Roadmap

```text
Data Science
     │
     ▼
Supervised Learning
     │
     ├── Regression
     ├── Regularization
     └── Classification
     │
     ▼
Unsupervised Learning
     │
     ├── Clustering
     └── Dimensionality Reduction
     │
     ▼
Deep Learning
     │
     ├── Perceptrons
     ├── Neural Networks
     ├── Backpropagation
     ├── PyTorch
     └── ANN Projects
     │
     ▼
Advanced AI
     │
     ├── CNNs
     ├── Transformers
     ├── Generative AI
     ├── RAG
     ├── AI Agents
     └── Research
```

The repository is currently moving from **classical Machine Learning and Data Science into Deep Learning**.

---

# 📚 What I Have Covered

## 📊 Data Science

### Pandas

* DataFrames
* Dataset exploration
* Searching and filtering
* Working with structured data
* Missing values
* Data manipulation

📁 `03-data-science/pandas/`

### Matplotlib

* Basic plotting
* Data visualization
* Charts and figures

📁 `03-data-science/matplotlib/`

### Datasets

Currently used datasets include:

* Pokémon dataset
* House-price datasets
* Regression practice datasets
* Customer churn dataset

📁 `03-data-science/data-bases/`

---

# 🤖 Machine Learning

## 📈 Regression

### Linear Regression

Concepts covered:

* Linear Regression
* Features and targets
* Coefficients
* Intercept
* Predictions
* Best-fit line
* Ordinary Least Squares
* Multiple Linear Regression

📁 `01-machine-learning/01-linear-regression/`

### Train/Test Split

* Training data
* Testing data
* Generalization
* `train_test_split`
* `test_size`
* `random_state`
* Evaluating models on unseen data

📁 `01-machine-learning/02-train-test-split/`

### Regression Metrics

* MAE
* MSE
* RMSE
* R² Score

📁 `01-machine-learning/03-regression-metrics/`

---

## 🧲 Regularization

### Ridge Regression

* Overfitting
* L2 regularization
* Regularization penalty
* Coefficient shrinkage
* `alpha`
* Implementation with Scikit-learn
* Implementation from scratch

📁 `01-machine-learning/04-ridge-regression/`

### Lasso Regression

* L1 regularization
* Overfitting
* Coefficient shrinkage
* Feature selection
* Coefficients becoming zero

📁 `01-machine-learning/05-lasso-regression/`

### Elastic Net

* L1 + L2 regularization
* Ridge + Lasso
* `alpha`
* `l1_ratio`
* Feature selection
* Correlated features

📁 `01-machine-learning/06-elastic-net/`

---

# 🎯 Classification

### Logistic Regression

* Classification vs Regression
* Binary classification
* Linear combination
* `z = wx + b`
* Sigmoid function
* Probability prediction
* Classification thresholds

📁 `01-machine-learning/07-logistic-regression/`

### Classification Metrics

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1 Score
* Log Loss
* Model Complexity

📁 `01-machine-learning/08-classification-metrics/`

### Decision Trees

* Decision tree intuition
* Decision rules
* Splitting data
* Classification
* Feature-based decisions
* Model interpretation

📁 `01-machine-learning/09-decision-tree/`

### Random Forest

* Ensemble learning
* Multiple decision trees
* Bootstrap sampling
* Feature randomness
* Classification

📁 `01-machine-learning/10-random-forest/`

### K-Nearest Neighbors

* Distance-based classification
* Nearest neighbors
* Choosing K
* Classification

📁 `01-machine-learning/11-knn/`

### Naive Bayes

* Bayes theorem
* Conditional probability
* Probabilistic classification

📁 `01-machine-learning/12-naive-bayes/`

### Support Vector Machines

* Hyperplanes
* Margins
* Support vectors
* Kernel intuition
* Classification

📁 `01-machine-learning/13-support-vector-machines/`

### Cross-Validation

* K-Fold Cross-Validation
* Validation strategy
* Model evaluation
* Generalization

📁 `01-machine-learning/14-cross-validation/`

### Hyperparameter Tuning

* Grid Search
* Hyperparameters
* Model comparison
* Selecting better configurations

📁 `01-machine-learning/15-hyperparameter-tuning/`

---

# 🔍 Unsupervised Learning

## K-Means Clustering

* Clustering
* Centroids
* Distance
* Iterative optimization
* Elbow Method

📁 `02-unsupervised-learning/01-k-means/`

## PCA

* Dimensionality reduction
* Principal components
* Variance
* Feature transformation

📁 `02-unsupervised-learning/02-pca/`

---

# 🧠 Deep Learning

I have now started learning Deep Learning.

The current focus is on understanding neural networks from the fundamentals rather than treating them as a black box.

## Perceptron

Topics:

* What a perceptron is
* Inputs
* Weights
* Bias
* Weighted sum
* Activation
* Prediction
* Learning intuition

📁 `02-deep-learning/01-Perceptron/`

---

# 🔥 Current Project — Customer Churn Prediction using ANN

I have started applying Deep Learning concepts by building a **Customer Churn Prediction model using an Artificial Neural Network (ANN)**.

📁 `projects/customer-churn-prediction-ann/`

The project is implemented in **PyTorch**, even though the original tutorial that inspired it used TensorFlow/Keras.

## What the project does

The model takes customer information such as:

* Age
* Gender
* Tenure
* Usage Frequency
* Support Calls
* Payment Delay
* Subscription Type
* Contract Length
* Total Spend
* Last Interaction

and predicts whether the customer is likely to churn.

---

## 🧠 ANN Architecture

The current network is:

```text
15 Input Features
       ↓
Linear 15 → 16
       ↓
ReLU
       ↓
Linear 16 → 8
       ↓
ReLU
       ↓
Linear 8 → 1
       ↓
Output
```

The model contains **401 trainable parameters**.

---

## ⚙️ PyTorch Concepts Used

This project introduced several important PyTorch concepts:

* `torch.Tensor`
* `nn.Module`
* `nn.Linear`
* `nn.ReLU`
* `BCEWithLogitsLoss`
* Adam optimizer
* Forward propagation
* Backpropagation
* Gradients
* `TensorDataset`
* `DataLoader`
* Mini-batch training
* GPU acceleration
* `model.train()`
* `model.eval()`
* `torch.no_grad()`
* `state_dict()`

---

## 🔄 Training Pipeline

The training process is:

```text
Dataset
   ↓
Preprocessing
   ↓
Train/Test Split
   ↓
Scaling
   ↓
PyTorch Dataset
   ↓
DataLoader
   ↓
Batch
   ↓
Forward Pass
   ↓
Loss
   ↓
Backpropagation
   ↓
Optimizer
   ↓
Updated Weights
   ↓
Repeat
```

This helped me understand what is actually happening behind a high-level:

```python
model.fit(X, y)
```

type of workflow.

---

## 💾 Model Persistence

The trained model can be saved and loaded later.

```text
churn_model.pth
```

stores the trained neural-network parameters.

```text
scaler.pkl
```

stores the fitted feature scaler.

This allows the model to be trained once and then used later for inference without retraining.

---

## 🔮 Inference

The project also contains a prediction workflow where a new customer's information can be entered manually.

The system:

```text
New Customer
     ↓
Preprocessing
     ↓
Scaling
     ↓
Trained ANN
     ↓
Sigmoid
     ↓
Churn Probability
     ↓
Churn / No Churn
```

Example:

```text
Churn Probability: 98.72%

Prediction: CUSTOMER WILL CHURN
```

---

# 🧪 Model Evaluation

Models are not judged only by whether they produce predictions.

I evaluate them using:

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1 Score
* Loss

For the current churn project, one training run reached approximately **99.68% test accuracy**.

However, this result is being treated cautiously.

A very high score can sometimes indicate:

* Duplicate records
* Data leakage
* Train/test contamination
* Dataset-specific patterns
* An unusually easy dataset

So the next step is not simply to celebrate the 99.68%.

The next step is to **investigate why the model performs so well and whether it actually generalizes**.

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
          Write Code
              │
              ▼
       Experiment with Data
              │
              ▼
        Evaluate Results
              │
              ▼
       Document What I Learned
              │
              ▼
       Find What I Got Wrong
              │
              ▼
          Improve It
```

The objective is to move beyond:

```python
model.fit(X, y)
model.predict(X)
```

and understand what happens **inside the model**.

---

# 🛠️ Tools & Technologies

| Technology   | Purpose                    |
| ------------ | -------------------------- |
| Python       | Programming                |
| NumPy        | Numerical computing        |
| Pandas       | Data manipulation          |
| Matplotlib   | Data visualization         |
| Scikit-learn | Classical Machine Learning |
| PyTorch      | Deep Learning              |
| Torchinfo    | Neural network summaries   |
| CUDA         | GPU acceleration           |
| Git          | Version control            |
| GitHub       | Code and documentation     |

---

# 📈 Progress

## ✅ Completed

### Data Science

* Pandas
* Matplotlib
* Dataset exploration
* Data cleaning

### Supervised Learning

* Linear Regression
* Polynomial Regression
* Train/Test Split
* MAE
* MSE
* RMSE
* R² Score
* Ridge Regression
* L2 Regularization
* Overfitting
* Lasso Regression
* L1 Regularization
* Feature Selection
* Elastic Net
* Logistic Regression
* Sigmoid Function
* Binary Classification
* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1 Score
* Log Loss
* Model Complexity
* Decision Trees
* Random Forest
* K-Nearest Neighbors
* Naive Bayes
* Support Vector Machines
* Cross-Validation
* Hyperparameter Tuning

### Unsupervised Learning

* K-Means Clustering
* Elbow Method
* PCA

### Deep Learning

* Perceptron
* Neural Network fundamentals
* PyTorch fundamentals
* ANN
* Forward Propagation
* Backpropagation
* Loss Functions
* Optimizers
* DataLoader
* Mini-batch Training
* Model Saving
* Model Loading
* Inference

---

# 🔨 Currently Learning

Deep Learning is currently the next major stage.

Upcoming topics include:

* Neural Networks from Scratch
* Activation Functions
* Backpropagation in greater depth
* Optimization
* PyTorch
* CNNs
* Image Classification
* Regularization in Neural Networks
* Dropout
* Batch Normalization

After the foundations are stronger, the journey will continue toward:

* NLP
* Transformers
* Generative AI
* RAG
* AI Agents
* Multimodal AI
* AI Research

---

# 🎯 Why I'm Building This

A lot of Machine Learning tutorials teach you how to use an API.

That's useful, but it's not enough for me.

I want to understand:

* What problem does the algorithm solve?
* What assumptions does it make?
* What is happening mathematically?
* How does the algorithm learn?
* What parameters control it?
* Why does it make a particular prediction?
* How can it overfit?
* How should it be evaluated?
* When should it be used?
* When should it not be used?

This repository is my attempt to answer those questions through:

```text
Code
  +
Mathematics
  +
Experiments
  +
Failures
  +
Debugging
  +
Documentation
```

---

# 🤖 AI-Assisted Learning

AI tools are part of this learning process.

**ChatGPT and other AI tools may be used to:**

* Explain difficult concepts
* Help translate implementations between frameworks
* Debug errors
* Suggest approaches
* Review code
* Help document projects
* Explain unexpected results

AI assistance is **not treated as a replacement for understanding**.

When AI helps write code, the goal is to understand the code, test it, break it, modify it, and eventually be able to explain what it is doing.

This repository therefore documents both the **learning process and the use of AI as a development/learning tool**.

---

# 🚧 Status

This repository is actively being developed.

I'm learning one concept at a time, implementing it, experimenting with datasets, debugging problems, evaluating results, and documenting what I understand.

The structure will change as the journey moves from:

```text
Data Science
      ↓
Machine Learning
      ↓
Unsupervised Learning
      ↓
Deep Learning
      ↓
Advanced AI
      ↓
Research
```

---

# 🤝 Let's Build Shit Together

This isn't a finished textbook.

It's a record of learning AI from the fundamentals.

Some code will be imperfect.

Some experiments will fail.

Some results will be suspicious.

Some implementations will eventually be rewritten.

That's the point.

If something is wrong, inefficient, outdated, or poorly implemented, that's part of the journey.

**Keep learning.
Keep building.
Keep breaking things.
Keep figuring out why they broke.**

⭐ Repository:
https://github.com/psorigins10/artificial-intelligence-learning-journey
