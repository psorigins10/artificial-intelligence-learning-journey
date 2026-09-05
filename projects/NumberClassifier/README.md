# ✏️ Number Classifier

> **An end-to-end handwritten digit recognition system built with PyTorch, FastAPI, and Flutter.**

Number Classifier started as a simple **MNIST modeling exercise** and evolved into a complete machine-learning application with a real client-server pipeline.

You can draw a handwritten digit directly on your phone or upload a PNG, send the image to a FastAPI backend, run it through a trained PyTorch model, and receive the predicted digit back in real time.

The interesting part of this project isn't just the classifier — it's the complete journey from **training a neural network → exposing it as an API → building a mobile client → connecting everything together.**

---

## 🧠 What This Project Does

The application recognizes handwritten digits from **0–9**.

There are two ways to provide an input:

* ✍️ **Draw** a digit using your finger
* 🖼️ **Upload** a PNG containing a digit

The complete pipeline looks like this:

```text
                 ┌──────────────────────┐
                 │      Flutter App     │
                 │                      │
                 │  Draw Digit / Upload │
                 └──────────┬───────────┘
                            │
                            │ PNG
                            ▼
                 ┌──────────────────────┐
                 │     FastAPI API      │
                 │                      │
                 │    POST /predict     │
                 └──────────┬───────────┘
                            │
                            │ PIL Image
                            ▼
                 ┌──────────────────────┐
                 │    Preprocessing     │
                 │                      │
                 │ Grayscale → Resize  │
                 │       → Tensor      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │    PyTorch MLP       │
                 │                      │
                 │   784 → 128 → 64    │
                 │         → 10         │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │      Prediction      │
                 │                      │
                 │        0 – 9         │
                 └──────────────────────┘
```

---

# 🏗️ Project Structure

```text
NumberClassifier/
│
├── ml/
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── best_mnist_mlp.pth
│
├── api/
│   └── main.py
│
├── frontend/
│   └── number_classifier_app/
│       ├── lib/
│       │   └── main.dart
│       ├── android/
│       ├── ios/
│       ├── pubspec.yaml
│       └── ...
│
├── data/
│   └── MNIST dataset
│
└── README.md
```

### `ml/`

Contains the machine-learning side of the project.

* `model.py` — defines the neural-network architecture
* `train.py` — downloads MNIST, trains the model, evaluates it, and saves the best checkpoint
* `predict.py` — loads the trained model and performs inference
* `best_mnist_mlp.pth` — saved PyTorch model weights

### `api/`

Contains the FastAPI backend.

* `main.py` exposes the `/predict` endpoint
* Receives uploaded images
* Validates them
* Converts them into PIL images
* Sends them to the ML inference function
* Returns the prediction as JSON

### `frontend/`

Contains the Flutter mobile application.

The app provides the user interface for drawing and uploading digits and communicates with the FastAPI backend over HTTP.

### `data/`

Contains the downloaded MNIST dataset used during training.

---

# 🤖 Machine Learning Model

The classifier is a fully-connected **Multi-Layer Perceptron (MLP)** trained on MNIST.

Architecture:

```text
Input
784 values
(28 × 28 pixels)
       │
       ▼
Linear
784 → 128
       │
       ▼
ReLU
       │
       ▼
Linear
128 → 64
       │
       ▼
ReLU
       │
       ▼
Linear
64 → 10
       │
       ▼
10 digit classes
0 1 2 3 4 5 6 7 8 9
```

The model contains:

* Input size: **784**
* Hidden layer 1: **128 neurons**
* Hidden layer 2: **64 neurons**
* Output layer: **10 neurons**
* Activation: **ReLU**
* Loss: **CrossEntropyLoss**
* Optimizer: **Adam**
* Learning rate: **0.001**
* Batch size: **64**
* Training: **20 epochs**

The model was trained using CUDA when available, including an NVIDIA GeForce RTX 4050 Laptop GPU during development.

---

# 📚 Why MNIST?

MNIST is a classic handwritten-digit dataset containing grayscale images of digits from **0 to 9**.

Each image is:

```text
28 × 28 pixels
```

which gives:

```text
28 × 28 = 784
```

input values.

MNIST is small, clean, and standardized, making it an excellent dataset for learning the fundamentals of neural networks and image classification.

However, there is an important catch.

**MNIST accuracy does not automatically translate into real-world handwriting accuracy.**

That distinction becomes especially important in this project because the model is being fed images coming from a phone rather than perfectly standardized MNIST samples.

---

# 📱 Flutter Application

The mobile frontend is built with **Flutter/Dart**.

The UI was designed as a dark, glassmorphic interface with neon gradient accents.

## Draw Mode

Users can draw a digit directly using their finger.

The drawing canvas includes:

* Smooth stroke rendering
* Rounded stroke caps
* Neon glow effect
* Dark canvas background
* Clear button
* Animated prediction state

The drawing is captured as a PNG before being sent to the backend.

## Upload Mode

Users can also provide an existing image.

Supported input methods include:

* 📱 Gallery
* 📷 Camera
* 🖼️ PNG image upload

The selected image is displayed inside the application before prediction.

---

# 🚀 FastAPI Backend

The backend is built using **FastAPI** and served with **Uvicorn**.

The primary endpoint is:

```text
POST /predict
```

It accepts an uploaded image using multipart form data.

The backend then:

1. Reads the uploaded bytes
2. Checks the file size
3. Opens the image using Pillow
4. Validates the image dimensions
5. Converts the image for model inference
6. Runs the PyTorch model
7. Returns the predicted digit

Example successful response:

```json
{
  "prediction": 8
}
```

---

# 🔒 Input Validation

The backend currently performs basic validation before inference.

### File size

Maximum:

```text
5 MB
```

### Image dimensions

Maximum:

```text
1920 × 1080
```

### Invalid image

If the uploaded file cannot be interpreted as an image, the API returns an error instead of passing it to the model.

These checks prevent obviously invalid or unnecessarily large inputs from reaching the inference pipeline.

---

# 🔌 Connecting Flutter to FastAPI

During local development, the Flutter application needs to know where the FastAPI server is running.

The API URL depends on the device:

| Environment                        | API URL                                    |
| ---------------------------------- | ------------------------------------------ |
| Android Emulator                   | `http://10.0.2.2:8000`                     |
| iOS Simulator                      | `http://127.0.0.1:8000`                    |
| Physical Android device over Wi-Fi | `http://<LAN-IP>:8000`                     |
| Physical Android device over USB   | `http://127.0.0.1:8000` with `adb reverse` |
| Deployed backend                   | `https://your-domain.com`                  |

For a physical Android device connected through USB, port forwarding can be used:

```bash
adb reverse tcp:8000 tcp:8000
```

For Wi-Fi development, both the phone and computer need to be connected to the same network.

---

# ▶️ Running Locally

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd NumberClassifier
```

## 2. Create/activate the Python environment

```bash
python -m venv .venv
```

Activate it according to your operating system.

## 3. Install backend dependencies

```bash
cd api
pip install -r requirements.txt
```

## 4. Start FastAPI

```bash
PYTHONPATH=..:../ml uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available locally at:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## 5. Run Flutter

From the Flutter project:

```bash
flutter pub get
flutter run
```

Make sure the API URL inside the Flutter application points to the correct FastAPI address for your device.

---

# 🧪 Testing

The backend can be tested independently through FastAPI's interactive Swagger UI:

```text
/docs
```

The `/predict` endpoint can be used to upload a PNG and verify the complete inference pipeline before connecting the mobile application.

This is useful because it separates backend problems from Flutter/networking problems.

---

# ✅ What Works

The current project successfully demonstrates:

* [x] MNIST dataset loading
* [x] PyTorch MLP architecture
* [x] Model training
* [x] Model evaluation
* [x] Best-model checkpoint saving
* [x] CUDA inference when available
* [x] Standalone prediction function
* [x] FastAPI backend
* [x] Multipart image uploads
* [x] Image validation
* [x] Pillow image processing
* [x] `/predict` API endpoint
* [x] Flutter mobile frontend
* [x] Finger drawing
* [x] Smooth/neon drawing UI
* [x] PNG upload
* [x] Gallery/camera input
* [x] Drawing → PNG conversion
* [x] Flutter → FastAPI communication
* [x] FastAPI → PyTorch inference
* [x] Prediction returned to the client
* [x] Error messages surfaced in the UI

The backend has been tested successfully with repeated `POST /predict` requests returning:

```text
200 OK
```

---

# ⚠️ Known Limitations

This project works, but it is important to be honest about where it is currently weak.

## 1. The model is an MLP, not a CNN

The biggest ML limitation is the model architecture.

The MLP works well as a learning project and performs reasonably on MNIST, but it does not exploit the spatial structure of images the way a CNN does.

For example:

```text
MLP:
pixels → fully connected layers → prediction

CNN:
pixels → convolutional features → spatial patterns → prediction
```

A CNN would generally be a better architecture for image classification.

---

## 2. MNIST vs real-world input

This is probably the most important limitation of the entire project.

MNIST images are:

* 28×28
* grayscale
* centered
* standardized
* relatively clean

A phone drawing or photograph can be:

* off-center
* too small
* too large
* too thick
* too thin
* noisy
* rotated
* poorly contrasted

So a model can perform very well on the MNIST test set while performing noticeably worse on actual phone inputs.

That isn't necessarily a bug in PyTorch.

It is a **data distribution problem**.

---

## 3. No confidence score

The API currently returns one predicted digit.

For example:

```json
{
  "prediction": 8
}
```

The model will always select one of the ten classes.

Even if the input is a random doodle, the model will still choose:

```text
0–9
```

There is currently no:

```text
"I don't know"
```

state.

---

## 4. Single-digit only

The application is designed for one digit at a time.

It does not currently support:

```text
12345
```

or:

```text
42
```

as multi-digit numbers.

It is a **single handwritten digit classifier**, not a general OCR system.

---

## 5. Real-world preprocessing is limited

The current preprocessing performs basic image conversion and resizing.

It does not yet automatically:

* detect the digit's bounding box
* crop empty space
* center the digit
* normalize stroke thickness
* deskew the image
* normalize contrast

This is one of the biggest opportunities for improving real-world accuracy without immediately changing the model.

---

## 6. Backend is not production-hardened

The current FastAPI backend is designed primarily as a learning project and demo.

It does not currently include:

* Authentication
* Rate limiting
* Production monitoring
* Request quotas
* Advanced logging
* Configurable validation limits
* Robust abuse protection

It should **not be exposed directly to the public internet indefinitely in its current development configuration.**

---

## 7. Synchronous inference

Inference currently happens directly inside the request flow.

That is perfectly reasonable for this small model and a learning project, but under significant concurrent traffic, the architecture would need to be improved.

Possible future solutions include:

* thread pools
* background workers
* dedicated inference services
* asynchronous job queues

---

## 8. Hardcoded API URL

The Flutter application currently uses a configured API URL.

Changing between:

```text
localhost
LAN IP
production server
```

requires changing the configuration.

A better production setup would use build-time configuration such as:

```text
--dart-define
```

or another environment configuration mechanism.

---

# 🧠 Why an MLP Instead of a CNN?

The MLP was a deliberate starting point.

MNIST is simple enough that an MLP can learn the task without requiring a complicated architecture.

That makes it useful for understanding:

* tensors
* layers
* forward propagation
* loss functions
* backpropagation
* optimizers
* training loops
* evaluation
* model checkpoints
* inference

The weakness becomes much more obvious when real-world images enter the system.

That actually makes this project more interesting.

The application demonstrates the difference between:

> **"My model works on the dataset."**

and:

> **"My model works on the actual data my users give it."**

Those are two very different engineering problems.

---

# 🔮 Future Improvements

There are several clear directions for improving the project.

## 🧠 Better model

Replace the MLP with a small CNN.

Possible architecture:

```text
Input
  ↓
Convolution
  ↓
ReLU
  ↓
Pooling
  ↓
Convolution
  ↓
ReLU
  ↓
Pooling
  ↓
Fully Connected
  ↓
10 classes
```

This should improve image recognition performance and make the model more suitable for image-based input.

---

## 🎯 Better preprocessing

Add a server-side preprocessing pipeline that:

1. Detects the non-background region
2. Crops the digit
3. Centers it
4. Resizes it while preserving aspect ratio
5. Normalizes contrast
6. Produces an MNIST-like 28×28 representation

This could potentially improve real-world performance substantially without changing the model architecture.

---

## 📊 Confidence and Top-K Predictions

Instead of returning only:

```json
{
  "prediction": 8
}
```

return something like:

```json
{
  "prediction": 8,
  "confidence": 0.94
}
```

or even the top three predictions.

This would make the application more informative when the model is uncertain.

---

## 🚫 Unknown Input Detection

Eventually, the application could recognize that an input is probably not a digit at all.

For example:

```text
User draws:
     ❤️

Model:
     "Not a digit"
```

This would require more than simply taking the highest-scoring class from the current ten-class model.

---

## 🌐 Production Deployment

The backend can eventually be deployed to a cloud platform and exposed through HTTPS.

The architecture would then become:

```text
Flutter App
     │
     │ HTTPS
     ▼
Public API
     │
     ▼
FastAPI
     │
     ▼
PyTorch Model
     │
     ▼
Prediction
```

The laptop would no longer need to stay online.

---

## 🔐 Production Security

Before exposing the API publicly, the project should eventually add:

* HTTPS
* Authentication/API keys where appropriate
* Rate limiting
* Request size limits
* Better error handling
* Logging
* Monitoring
* Environment-based configuration

---

# 📦 Dependencies

### Machine Learning

* PyTorch
* torchvision
* MNIST

### Backend

* FastAPI
* Uvicorn
* Pillow
* python-multipart

### Frontend

* Flutter
* Dart
* `http`
* `image_picker`

---

# 🎓 What I Learned From This Project

This project started with a neural network and ended up teaching something much bigger than model training.

The project covers the complete path:

```text
Dataset
   ↓
Training
   ↓
Model
   ↓
Checkpoint
   ↓
Inference
   ↓
API
   ↓
Network
   ↓
Mobile Application
   ↓
Real User Input
```

The biggest lesson is that **machine learning engineering doesn't end when the model reaches a good test accuracy.**

A usable ML system also needs:

* input handling
* preprocessing
* inference
* APIs
* networking
* frontend integration
* validation
* error handling
* deployment
* security
* monitoring

That's what this project is ultimately about.

---

# 🚀 Project Status

**Current status: Working end-to-end prototype**

```text
Model Training       ✅
Model Inference      ✅
FastAPI Backend      ✅
Image Upload         ✅
Flutter Frontend     ✅
Drawing Input        ✅
API Communication    ✅
Mobile Testing       ✅
Production Deploy    ⏳
Production Security  ⏳
CNN Upgrade          ⏳
Advanced Preprocess  ⏳
Confidence Scores    ⏳
```

---

## ⭐ Final Note

This isn't intended to be presented as a state-of-the-art digit recognition system.

It's a practical **end-to-end machine-learning project** built from the ground up.

The MLP is intentionally simple.
The backend is intentionally lightweight.
The frontend is intentionally focused.

The goal was to take a model from:

```text
"it works in a Python notebook"
```

to:

```text
"someone can draw a digit on a phone and get a prediction from a real API."
```

And that's the part that makes this project worth building.
