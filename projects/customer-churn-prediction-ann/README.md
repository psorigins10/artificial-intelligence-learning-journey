# Customer Churn Prediction using ANN — PyTorch

A customer churn prediction project built with **PyTorch** using an Artificial Neural Network (ANN).

The goal of this project is to take customer information such as age, tenure, usage frequency, support calls, payment delays, subscription type, contract length, total spend, and last interaction, and predict whether the customer is likely to churn.

> **AI Assistance Disclosure:** AI (ChatGPT) was used during the development of this project to help explain concepts, translate the original TensorFlow/Keras tutorial approach into PyTorch, write and modify code, troubleshoot errors, and understand the training and prediction pipeline. The model was trained and executed locally using PyTorch.

---

## Project Overview

This project follows a complete machine-learning workflow:

```text
Customer Dataset
       ↓
Data Cleaning
       ↓
Feature Selection
       ↓
Categorical Encoding
       ↓
Train/Test Split
       ↓
Feature Scaling
       ↓
PyTorch Tensors
       ↓
Dataset + DataLoader
       ↓
Artificial Neural Network
       ↓
Loss Calculation
       ↓
Backpropagation
       ↓
Adam Optimizer
       ↓
Model Evaluation
       ↓
Save Trained Model
       ↓
Load Model Later
       ↓
Predict Churn for New Customers
````

---

## Dataset

The dataset used in this project is:

`customer_churn_dataset-training-master.csv`

The dataset contains customer information including:

* `CustomerID`
* `Age`
* `Gender`
* `Tenure`
* `Usage Frequency`
* `Support Calls`
* `Payment Delay`
* `Subscription Type`
* `Contract Length`
* `Total Spend`
* `Last Interaction`
* `Churn`

The `CustomerID` column is removed because it is an identifier and does not provide useful predictive information.

Missing rows are removed before training.

---

## Data Preprocessing

### 1. Remove Missing Values

Rows containing missing values are removed:

```python
df = df.dropna()
```

### 2. Separate Features and Target

`Churn` is the target variable.

```python
X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]
```

### 3. One-Hot Encoding

Categorical features are converted into numerical features:

```python
categorical_cols = [
    "Gender",
    "Subscription Type",
    "Contract Length"
]

X = pd.get_dummies(
    X,
    columns=categorical_cols,
    dtype=int
)
```

After encoding, the model receives **15 input features**.

### 4. Train/Test Split

The dataset is split into:

* 80% training data
* 20% testing data

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### 5. Feature Scaling

`StandardScaler` is fitted only on the training data:

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

The same fitted scaler is later saved and used when making predictions on new customers.

---

## Neural Network Architecture

The ANN is implemented using PyTorch and object-oriented programming.

Architecture:

```text
15 Input Features
       ↓
Linear Layer: 15 → 16
       ↓
ReLU
       ↓
Linear Layer: 16 → 8
       ↓
ReLU
       ↓
Linear Layer: 8 → 1
       ↓
Raw Output / Logit
```

The model contains **401 trainable parameters**.

The final layer does not contain a Sigmoid activation because the project uses:

```python
nn.BCEWithLogitsLoss()
```

Sigmoid is applied during prediction to convert the raw output into a probability.

---

## Training

Training uses PyTorch's `Dataset` and `DataLoader`.

Instead of sending the entire dataset through the model at once, the training data is divided into mini-batches.

Current batch size:

```python
batch_size = 64
```

The training flow for each batch is:

```text
Batch of Customers
       ↓
Model Forward Pass
       ↓
Predictions
       ↓
Calculate Loss
       ↓
optimizer.zero_grad()
       ↓
loss.backward()
       ↓
optimizer.step()
```

### What Happens During Training?

#### Forward Propagation

The customer's features pass through the neural network and produce a raw prediction.

#### Loss Calculation

`BCEWithLogitsLoss` compares the prediction with the actual churn label.

#### Backpropagation

```python
loss.backward()
```

PyTorch calculates gradients showing how each trainable parameter contributed to the error.

#### Optimizer Update

```python
optimizer.step()
```

The Adam optimizer uses those gradients to update the model's weights.

This process repeats for every batch and every epoch.

---

## Optimizer

The project uses the Adam optimizer:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

Adam adjusts the neural network's weights based on the gradients calculated during backpropagation.

---

## GPU Support

The project automatically uses an NVIDIA GPU when CUDA is available:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
```

Otherwise, it falls back to the CPU.

The model was trained and executed using a CUDA-enabled NVIDIA GPU environment.

---

## Model Evaluation

The trained model is evaluated on the test set.

During evaluation:

```python
probability = torch.sigmoid(output)
```

converts the model's raw output into a probability between 0 and 1.

A threshold of `0.5` is used:

```python
prediction = (probability >= 0.5).float()
```

Therefore:

```text
Probability < 0.5 → No Churn
Probability ≥ 0.5 → Churn
```

The project evaluates the model using:

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1-score

An earlier training run produced approximately **97% test accuracy**. After switching to mini-batch training, a later run reached approximately **99.68% test accuracy**.

Because the later result is unusually high, it should be treated as a result to investigate rather than automatically assuming it represents real-world performance. Duplicate records, dataset characteristics, and possible data leakage should be checked before considering the model production-ready.

---

## Saving the Model

After training, the learned model parameters are saved:

```python
torch.save(
    model.state_dict(),
    "churn_model.pth"
)
```

The fitted scaler is also saved:

```python
joblib.dump(
    scaler,
    "scaler.pkl"
)
```

These files allow the trained model to be reused without training it again.

### Saved Files

```text
churn_model.pth
scaler.pkl
```

* `churn_model.pth` → trained neural-network weights
* `scaler.pkl` → fitted feature scaler

---

## Making Predictions Later

The project includes a prediction workflow that:

1. Loads the trained ANN.
2. Loads the saved scaler.
3. Accepts information for a new customer.
4. Applies the same preprocessing used during training.
5. Scales the new customer using the saved scaler.
6. Passes the customer through the trained ANN.
7. Converts the output into a churn probability.
8. Produces a churn / no-churn prediction.

Example:

```text
Enter customer information:

Age: 18
Gender: Male
Tenure: 5
Usage Frequency: 56
Support Calls: 4
Payment Delay: 13
Subscription Type: Premium
Contract Length: Monthly
Total Spend: 995
Last Interaction: 20
```

Example output:

```text
==============================
       CHURN PREDICTION
==============================

Churn Probability: 100.00%
Prediction: CUSTOMER WILL CHURN
```

The displayed `100.00%` should not be interpreted as absolute certainty. It is a rounded probability produced by the neural network.

---

## Project Structure

A suggested project structure is:

```text
customer-churn-prediction-ann/
│
├── main.py
├── predict.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── churn_model.pth
│   └── scaler.pkl
│
└── data/
    └── customer_churn_dataset-training-master.csv
```

Do not commit sensitive data or datasets that you do not have permission to redistribute.

---

## Technologies Used

* Python
* PyTorch
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Torchinfo
* CUDA
* NVIDIA GPU

---

## What I Learned

This project was used to understand the complete ANN workflow in PyTorch, including:

* Loading and cleaning a real dataset
* Handling categorical variables
* One-hot encoding
* Train/test splitting
* Feature scaling
* Converting data to PyTorch tensors
* Using `TensorDataset`
* Using `DataLoader`
* Building an ANN with `nn.Module`
* Understanding layers and parameters
* Forward propagation
* Loss functions
* Backpropagation
* Gradients
* Optimizers
* Mini-batch training
* GPU acceleration
* Model evaluation
* Saving and loading trained models
* Running inference on new data

---

## AI Assistance

This project was developed as a learning project with assistance from **AI (ChatGPT)**.

AI assistance was used for:

* Explaining machine-learning and neural-network concepts
* Translating a TensorFlow/Keras tutorial into PyTorch
* Writing and modifying PyTorch code
* Debugging preprocessing and training issues
* Explaining errors and unexpected results
* Building the model-saving and prediction workflow
* Helping document the project

The dataset preprocessing, model training, execution, evaluation, and prediction were performed in the local development environment.

The use of AI assistance is intentionally documented here for transparency.

---

## Disclaimer

This is an educational machine-learning project. The model's predictions should not be treated as guaranteed predictions of actual customer behavior.

Before using a churn model in a real business environment, additional validation should be performed, including checking for:

* Data leakage
* Duplicate records
* Train/test contamination
* Distribution differences between datasets
* Model performance on genuinely unseen data
* Appropriate business metrics
* Probability calibration
* Bias and fairness
* Model performance over time

---

## Status

**Completed — Educational ANN churn prediction pipeline**

The project currently supports:

* Model training
* Mini-batch training with PyTorch DataLoader
* GPU acceleration
* Model evaluation
* Saving trained model weights
* Saving preprocessing scaler
* Loading the trained model
* Predicting churn for manually entered customers

```
```