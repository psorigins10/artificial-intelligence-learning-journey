# 🤖 Artificial Intelligence Learning Journey

> **Learning AI from the fundamentals — mathematics → algorithms → code → experiments → real projects.**

This repository is my hands-on journey through **Data Science, Machine Learning, Deep Learning, and eventually Advanced AI**.

I'm not trying to just learn how to call APIs or copy implementations.

I want to understand:

* What the algorithm is doing
* Why it works
* What is happening mathematically
* How models actually learn
* How parameters are updated
* How models are evaluated
* Why models fail
* When an algorithm should be used
* When it should **not** be used

This repository contains the things I learn, build, break, debug, experiment with, and gradually understand.

**The goal is not to make the repository look perfect.**
**The goal is to make my understanding stronger.**

---

# 🧭 Learning Roadmap

```text
Data Science
     │
     ▼
Machine Learning
     │
     ├── Regression
     ├── Classification
     ├── Regularization
     ├── Model Evaluation
     └── Unsupervised Learning
             │
             ▼
       Deep Learning
             │
             ├── Perceptron
             ├── Neural Networks
             ├── Forward Propagation
             ├── Loss Functions
             ├── Backpropagation
             ├── Gradient Descent
             ├── PyTorch
             └── ANN Projects
                    │
                    ▼
              Computer Vision
                    │
                    ├── CNNs
                    └── Image Classification
                           │
                           ▼
                     Advanced AI
                           │
                           ├── NLP
                           ├── Transformers
                           ├── Generative AI
                           ├── RAG
                           ├── AI Agents
                           ├── Multimodal AI
                           └── AI Research
```

The repository is currently transitioning from **classical Machine Learning into Deep Learning**.

---

# 📚 What I've Learned So Far

## 📊 Data Science

### Pandas

Topics explored:

* DataFrames
* Dataset exploration
* Searching and filtering
* Missing values
* Data manipulation
* Working with structured data

📁 `03-data-science/pandas/`

### Matplotlib

* Basic plotting
* Data visualization
* Charts and figures
* Exploring datasets visually

📁 `03-data-science/matplotlib/`

### Datasets

I've worked with datasets including:

* Pokémon data
* House-price data
* Regression datasets
* Customer churn data

📁 `03-data-science/data-bases/`

---

# 🤖 Machine Learning

## 📈 Regression

### Linear Regression

Concepts:

* Features and targets
* Coefficients
* Intercept
* Predictions
* Best-fit line
* Ordinary Least Squares
* Multiple Linear Regression

📁 `01-machine-learning/01-linear-regression/`

### Train/Test Split

* Training vs testing data
* Generalization
* `train_test_split`
* `test_size`
* `random_state`
* Evaluating on unseen data

📁 `01-machine-learning/02-train-test-split/`

### Regression Metrics

* MAE
* MSE
* RMSE
* R²

📁 `01-machine-learning/03-regression-metrics/`

---

## 🧲 Regularization

### Ridge Regression

* Overfitting
* L2 regularization
* Regularization penalty
* Coefficient shrinkage
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

Topics covered:

### Logistic Regression

* Classification vs regression
* Binary classification
* Linear combination
* `z = wx + b`
* Sigmoid
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
* Model complexity

📁 `01-machine-learning/08-classification-metrics/`

### Decision Trees

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

This is where the journey is getting serious.

The current focus is understanding **neural networks from the inside**, rather than treating them as black boxes.

## Perceptron

Topics:

* Inputs
* Weights
* Bias
* Weighted sum
* Activation
* Prediction
* Learning intuition

📁 `02-deep-learning/01-Perceptron/`

---

# 🔥 Projects

Projects are where I take the concepts I've learned and try to turn them into actual working systems.

---

## 🧠 Customer Churn Prediction — ANN

📁 `projects/customer-churn-prediction-ann/`

A customer churn prediction system built using **PyTorch**.

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

### Architecture

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

The network contains **401 trainable parameters**.

### Concepts practiced

* `torch.Tensor`
* `nn.Module`
* `nn.Linear`
* `nn.ReLU`
* `BCEWithLogitsLoss`
* Adam optimizer
* Forward propagation
* Backpropagation
* Gradients
* TensorDataset
* DataLoader
* Mini-batch training
* GPU acceleration
* Training/evaluation modes
* Model saving
* Model loading
* Inference

The project also helped me understand the complete training loop:

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

---

# 🔢 Handwritten Digit Classifier

📁 `projects/NumberClassifier/`

This is my first **end-to-end Deep Learning application**.

The goal:

> Draw or upload a handwritten digit → send it to an API → run it through a trained neural network → return the predicted digit.

### System Architecture

```text
                 Flutter App
                     │
          Draw / Upload Digit
                     │
                     ▼
                PNG Image
                     │
                     ▼
                FastAPI API
                     │
                     ▼
             Image Preprocessing
                     │
                     ▼
              PyTorch MLP
                     │
                     ▼
              Prediction 0–9
                     │
                     ▼
                FastAPI
                     │
                     ▼
                Flutter UI
```

### Neural Network

The classifier uses an MLP:

```text
Input Image
28 × 28
   ↓
Flatten
   ↓
784
   ↓
Linear 784 → 128
   ↓
ReLU
   ↓
Linear 128 → 64
   ↓
ReLU
   ↓
Linear 64 → 10
   ↓
Digit Prediction
```

### Training

The model was trained on **MNIST** using:

* PyTorch
* Cross Entropy Loss
* Adam optimizer
* Learning rate: `0.001`
* Batch size: `64`
* 20 epochs
* CUDA/GPU acceleration when available

The trained model is saved as:

```text
best_mnist_mlp.pth
```

### Backend

The model is exposed through a **FastAPI** endpoint:

```text
POST /predict
```

The API:

* Accepts an uploaded image
* Validates the request
* Converts the image using PIL
* Preprocesses it to the model's expected format
* Runs inference
* Returns the predicted digit

### Frontend

The application is built with **Flutter**.

It supports:

* ✏️ Drawing a digit
* 🖼️ Uploading an image
* 📷 Camera input
* 🔮 Prediction results
* ⏳ Loading states
* ❌ Error handling
* 🌑 Dark/glassmorphic UI

This project was important because it moved beyond:

```text
Train model
   ↓
Print prediction
```

into:

```text
Machine Learning Model
        +
Backend API
        +
Mobile Application
        =
End-to-End AI System
```

---

# 🧮 Understanding Neural Networks

I'm currently moving deeper into the mathematics behind neural networks.

The goal is to understand what happens between:

```python
loss.backward()
optimizer.step()
```

rather than simply knowing that these lines are required.

Current concepts:

* Perceptrons
* Weighted sums
* Bias
* Activation functions
* Forward propagation
* Loss functions
* Gradients
* Chain rule
* Backpropagation
* Gradient descent
* Optimization
* Neural-network parameters

### Current focus

🔥 **Backpropagation**

The next goal is to understand it mathematically and implement the ideas from scratch rather than relying entirely on PyTorch's automatic differentiation.

---

# 🧪 Model Evaluation

A model producing a prediction does not automatically mean the model is good.

I'm learning to evaluate models using:

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1 Score
* Loss
* Generalization
* Train/test performance

I'm also learning to question unusually good results.

A high score can sometimes hide:

* Data leakage
* Duplicate records
* Train/test contamination
* Dataset-specific patterns
* Overfitting
* An unusually easy dataset

The goal is not:

> **"I got 99%, therefore my model is amazing."**

The goal is:

> **"Why did I get 99%, and does the model actually generalize?"**

---

# 📈 Current Progress

## ✅ Completed

### Data Science

* Pandas
* Matplotlib
* Dataset exploration
* Data cleaning

### Machine Learning

* Linear Regression
* Polynomial Regression
* Train/Test Split
* MAE
* MSE
* RMSE
* R²
* Ridge Regression
* L2 Regularization
* Lasso Regression
* L1 Regularization
* Elastic Net
* Logistic Regression
* Sigmoid
* Binary Classification
* Classification Metrics
* Decision Trees
* Random Forest
* KNN
* Naive Bayes
* SVM
* Cross-Validation
* Hyperparameter Tuning
* K-Means
* PCA

### Deep Learning

* Perceptron
* Neural Network fundamentals
* PyTorch fundamentals
* ANN
* Forward Propagation
* Loss Functions
* Optimizers
* DataLoader
* Mini-batch Training
* Model Saving
* Model Loading
* Inference
* GPU acceleration
* Building an end-to-end ML application
* 🧮 Backpropagation

---

# 🚧 Currently Learning

The current stage of the journey is **Deep Learning fundamentals**.

### Right now

* 🔗 Chain Rule
* 📉 Gradient Descent
* 🧠 Neural Networks from Scratch
* ⚡ Activation Functions
* 🔥 PyTorch internals
* Vanishing Gradient Problem

### Coming next

* CNNs
* Image Classification
* Regularization
* Dropout
* Batch Normalization
* Better preprocessing
* Model architectures
* Computer Vision

### Eventually

* NLP
* Transformers
* Generative AI
* RAG
* AI Agents
* Multimodal AI
* AI Research

---

# 🛠️ Tools & Technologies

| Technology   | Purpose                        |
| ------------ | ------------------------------ |
| Python       | Programming                    |
| NumPy        | Numerical computing            |
| Pandas       | Data manipulation              |
| Matplotlib   | Visualization                  |
| Scikit-learn | Classical Machine Learning     |
| PyTorch      | Deep Learning                  |
| Torchinfo    | Neural-network summaries       |
| CUDA         | GPU acceleration               |
| FastAPI      | ML API development             |
| Flutter      | Mobile application development |
| Git          | Version control                |
| GitHub       | Code & documentation           |

---

# 🧠 My Learning Process

For each topic, I try to follow this cycle:

```text
Learn the Concept
       ↓
Understand the Mathematics
       ↓
Understand the Algorithm
       ↓
Write Code
       ↓
Experiment
       ↓
Evaluate Results
       ↓
Document What I Learned
       ↓
Find What I Got Wrong
       ↓
Fix It
       ↓
Repeat
```

I'm deliberately trying to move beyond:

```python
model.fit(X, y)
model.predict(X)
```

and toward:

```text
What is happening inside the model?
```

---

# 🤖 AI-Assisted Learning

AI tools are part of this learning process.

I may use AI to:

* Explain difficult concepts
* Debug errors
* Review implementations
* Translate ideas between frameworks
* Suggest approaches
* Help document projects
* Explain unexpected results

But AI assistance is **not a replacement for understanding**.

If AI helps me write something, the goal is to:

1. Understand it
2. Run it
3. Break it
4. Modify it
5. Debug it
6. Explain it myself

The final goal is **understanding, not just working code**.

---

# 🎯 Why I'm Building This

A lot of Machine Learning education focuses on:

```text
Import library
      ↓
Call API
      ↓
Get prediction
      ↓
Celebrate accuracy
```

That's useful.

But I want to go deeper.

I want to understand:

* What problem does the algorithm solve?
* What assumptions does it make?
* What is happening mathematically?
* How does it learn?
* How are parameters updated?
* Why does it make a particular prediction?
* Why does it fail?
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
       Failure
         +
      Debugging
         +
    Documentation
         +
      Curiosity
```

---

# 🚧 This Repository Is Supposed to Be Imperfect

This isn't a finished textbook.

It's a record of learning.

Some code will be:

* Imperfect
* Rewritten
* Inefficient
* Experimental
* Wrong at first
* Eventually improved

Some experiments will fail.

Some results will look suspicious.

That's part of the process.

The repository will change as my understanding changes.

---

# 🗺️ Long-Term Direction

```text
Data Science
      ↓
Machine Learning
      ↓
Deep Learning
      ↓
Computer Vision
      ↓
NLP
      ↓
Transformers
      ↓
Generative AI
      ↓
AI Agents
      ↓
Multimodal AI
      ↓
AI Research
```

One concept at a time.

One experiment at a time.

One bug at a time.

---

# 🔥 Let's Build Shit Together

This repository is my learning journey.

Not a course.

Not a polished portfolio pretending everything worked perfectly.

Not a collection of copied notebooks.

It's the actual process:

**learn → build → break → debug → understand → improve.**

If something here is wrong, outdated, inefficient, or could be explained better, that's part of the journey.

**Keep learning.**
**Keep building.**
**Keep breaking things.**
**Keep figuring out why they broke.**

⭐ [Repository](https://github.com/psorigins10/artificial-intelligence-learning-journey)
