# MNIST MLP Digit Classifier

A small PyTorch project that trains a **Multi-Layer Perceptron (MLP)** to recognize handwritten digits from the MNIST dataset.

The project demonstrates the basic machine-learning workflow:

**Load data → Train model → Evaluate → Save best model → Load model → Predict new images**

---

## Project Overview

The goal of this project is to build a simple neural network that can classify images of handwritten digits from **0 to 9**.

We use the **MNIST dataset**, which contains:

* **60,000 training images**
* **10,000 test images**
* Images are **28 × 28 pixels**
* Images are grayscale
* There are **10 classes: 0–9**

Each image therefore contains:

```text
28 × 28 = 784 pixels
```

The MLP receives these 784 pixel values as its input.

---

## Model Architecture

The neural network is:

```text
784
 ↓
Linear(784 → 128)
 ↓
ReLU
 ↓
Linear(128 → 64)
 ↓
ReLU
 ↓
Linear(64 → 10)
```

In PyTorch:

```python
self.network = nn.Sequential(
    nn.Flatten(),

    nn.Linear(784, 128),
    nn.ReLU(),

    nn.Linear(128, 64),
    nn.ReLU(),

    nn.Linear(64, 10)
)
```

### Why 784 inputs?

MNIST images are 28 × 28 pixels:

```text
28 × 28 = 784
```

The `Flatten()` layer converts the 2D image into a 1D vector:

```text
[1, 28, 28]
       ↓
[784]
```

### Why 10 outputs?

MNIST contains ten possible digits:

```text
0 1 2 3 4 5 6 7 8 9
```

Therefore, the final layer produces 10 output values.

The largest output determines the predicted digit.

---

## Training

The model is trained using:

### Loss Function

```python
nn.CrossEntropyLoss()
```

Cross-entropy measures how far the model's predictions are from the correct labels.

### Optimizer

```python
torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

Adam updates the model's weights using the gradients calculated during backpropagation.

### Batch Size

```python
batch_size = 64
```

The model processes 64 images at a time instead of all 60,000 images simultaneously.

One epoch processes the complete training dataset.

Approximately:

```text
60,000 / 64 ≈ 938 batches per epoch
```

### Epochs

The model is trained for:

```python
epochs = 15
```

This means the training dataset is processed approximately 15 times.

---

## GPU Support

The project automatically uses an NVIDIA GPU when CUDA is available:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

Otherwise, it falls back to the CPU.

Example:

```text
Using device: cuda
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
```

Both the model and input tensors are moved to the selected device:

```python
model = MLP().to(device)

images = images.to(device)
labels = labels.to(device)
```

---

## Training Process

For every batch, the following process happens:

```text
Images
   ↓
Forward Pass
   ↓
Predictions
   ↓
Calculate Loss
   ↓
Zero Gradients
   ↓
Backpropagation
   ↓
Calculate Gradients
   ↓
Update Weights
```

The core training code is:

```python
prediction = model(images)

loss = loss_fn(prediction, labels)

optimizer.zero_grad()

loss.backward()

optimizer.step()
```

### `loss.backward()`

This calculates the gradients of the model's parameters.

### `optimizer.step()`

This uses those gradients to update the model's weights.

Over many batches and epochs, the model learns to recognize patterns associated with each digit.

---

## Saving the Best Model

Instead of automatically keeping the model from the final epoch, the project tracks the best accuracy:

```python
best_accuracy = 0.0
```

After every epoch, the model is evaluated.

If its accuracy is better than the previous best:

```python
if accuracy > best_accuracy:

    best_accuracy = accuracy

    torch.save(
        model.state_dict(),
        "best_mnist_mlp.pth"
    )
```

The trained weights are saved to:

```text
best_mnist_mlp.pth
```

This file contains the learned parameters of the neural network.

---

## Loading the Saved Model

A separate prediction script recreates the same MLP architecture:

```python
model = MLP().to(device)
```

Then loads the saved weights:

```python
model.load_state_dict(
    torch.load(
        "best_mnist_mlp.pth",
        map_location=device,
        weights_only=True
    )
)
```

The model is then switched to evaluation mode:

```python
model.eval()
```

At this point, the model is ready to make predictions without retraining.

---

## Image Prediction

For a new image, the image must first be converted into the same general format used by MNIST.

The preprocessing pipeline is:

```text
Input Image
    ↓
Grayscale
    ↓
Resize to 28 × 28
    ↓
Convert to Tensor
    ↓
Add Batch Dimension
    ↓
MLP
    ↓
Prediction
```

The preprocessing code:

```python
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])
```

For example, an image containing the digit:

```text
8
```

can be passed through the trained model and produce:

```text
Model prediction: 8
```

---

## Model Output

The final layer produces 10 values called **logits**:

```text
0 → score
1 → score
2 → score
...
8 → score
9 → score
```

The prediction is obtained using:

```python
prediction = output.argmax(dim=1).item()
```

`argmax()` selects the index with the largest value.

For example:

```text
Digit:   0    1    2    3    4    5    6    7    8    9
Score:  -3   -2   -1   -2   -4   -1   -5    1    9   -2
                                                   ↑
                                                largest
```

Therefore:

```text
Prediction = 8
```

---

## Results

The MLP achieved approximately:

```text
97.68% test accuracy
```

on an earlier 10-epoch training run.

This means that approximately:

```text
9,768 / 10,000
```

test images were classified correctly, while approximately:

```text
232 / 10,000
```

were classified incorrectly.

The final 15-epoch run should be evaluated separately rather than assuming it will produce exactly the same accuracy.

---

## Project Structure

A simple project structure is:

```text
mnist-mlp/
│
├── train.py
├── predict.py
├── best_mnist_mlp.pth
├── digit8.png
│
└── data/
    └── MNIST dataset files
```

### `train.py`

Downloads MNIST, creates the MLP, trains it, evaluates it, and saves the best model.

### `predict.py`

Loads the saved model and uses it to classify a new image.

### `best_mnist_mlp.pth`

Contains the learned model weights.

### `digit8.png`

Example external image used for inference.

### `data/`

Contains the downloaded MNIST dataset.

---

## Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy / PyTorch tensors
* PIL
* Matplotlib (for visualization)

---

## What This Project Demonstrates

This mini project covers the fundamentals of a PyTorch machine-learning workflow:

* Loading a dataset
* Using `Dataset` and `DataLoader`
* Working with tensors
* Understanding image shapes
* Building an MLP
* Using `nn.Linear`
* Using ReLU activation
* Forward propagation
* Cross-entropy loss
* Backpropagation
* Gradient descent
* Adam optimizer
* GPU acceleration with CUDA
* Model evaluation
* Saving model weights
* Loading model weights
* Image preprocessing
* Running inference on an external image

---

## Possible Next Steps

Some natural improvements to this project are:

1. Plot training loss and accuracy with Matplotlib.
2. Display correctly and incorrectly classified digits.
3. Show the model's confidence for all 10 classes.
4. Add a proper validation set instead of using the test set during model selection.
5. Test the model on handwritten images from a camera.
6. Compare the MLP against a CNN.
7. Experiment with different numbers of hidden layers and neurons.
8. Build a simple GUI where a user can draw a digit and get a prediction.

---

## Key Takeaway

This project demonstrates the complete basic neural-network workflow:

```text
DATA
 ↓
PREPROCESSING
 ↓
MODEL
 ↓
TRAINING
 ↓
LOSS
 ↓
BACKPROPAGATION
 ↓
WEIGHT UPDATES
 ↓
EVALUATION
 ↓
SAVE MODEL
 ↓
LOAD MODEL
 ↓
NEW IMAGE
 ↓
PREDICTION
```

The main purpose is not just to achieve high MNIST accuracy, but to understand what happens at each stage of the process.
