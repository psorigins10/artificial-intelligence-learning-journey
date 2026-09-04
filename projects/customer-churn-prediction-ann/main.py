import torch
import torch.nn as nn
import joblib
import pandas as pd

from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from torchinfo import summary


# ============================================================
# 1. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# 2. Load dataset
# ============================================================

df = pd.read_csv(
    "../../03-data-science/data-bases/customer_churn_dataset-training-master.csv"
)

# Remove rows containing missing values
df = df.dropna()


# ============================================================
# 3. Prepare X and y
# ============================================================

X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]


# ============================================================
# 4. Encode categorical columns
# ============================================================

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


# ============================================================
# 5. Train / Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 6. Feature Scaling
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ============================================================
# 7. Convert to PyTorch tensors
# ============================================================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train.values,
    dtype=torch.float32
).reshape(-1, 1)

y_test = torch.tensor(
    y_test.values,
    dtype=torch.float32
).reshape(-1, 1)


# ============================================================
# 8. Create Dataset
# ============================================================

train_dataset = TensorDataset(
    X_train,
    y_train
)

test_dataset = TensorDataset(
    X_test,
    y_test
)


# ============================================================
# 9. Create DataLoaders
# ============================================================

batch_size = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# ============================================================
# 10. Define Neural Network
# ============================================================

class ChurnANN(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(15, 16)
        self.layer2 = nn.Linear(16, 8)
        self.output_layer = nn.Linear(8, 1)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.layer1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.relu(x)

        x = self.output_layer(x)

        return x


# ============================================================
# 11. Create Model
# ============================================================

model = ChurnANN().to(device)


# ============================================================
# 12. Model Summary
# ============================================================

summary(
    model,
    input_size=(1, 15),
    device=device
)


# ============================================================
# 13. Loss Function
# ============================================================

loss_function = nn.BCEWithLogitsLoss()


# ============================================================
# 14. Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 15. Training
# ============================================================

epochs = 30

for epoch in range(epochs):

    model.train()

    total_loss = 0.0

    for X_batch, y_batch in train_loader:

        # Move batch to GPU
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # --------------------------------------------
        # Forward propagation
        # --------------------------------------------

        predictions = model(X_batch)

        # --------------------------------------------
        # Calculate loss
        # --------------------------------------------

        loss = loss_function(
            predictions,
            y_batch
        )

        # --------------------------------------------
        # Clear old gradients
        # --------------------------------------------

        optimizer.zero_grad()

        # --------------------------------------------
        # Backpropagation
        # --------------------------------------------

        loss.backward()

        # --------------------------------------------
        # Update weights
        # --------------------------------------------

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {average_loss:.4f}"
    )


# ============================================================
# 16. Evaluation
# ============================================================

model.eval()

all_predictions = []
all_targets = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(device)

        predictions = model(X_batch)

        probabilities = torch.sigmoid(predictions)

        predicted_classes = (
            probabilities >= 0.5
        ).float()

        all_predictions.append(
            predicted_classes.cpu()
        )

        all_targets.append(
            y_batch
        )


# Combine all batches
predicted_classes = torch.cat(
    all_predictions
)

y_test_cpu = torch.cat(
    all_targets
)


# ============================================================
# 17. Accuracy
# ============================================================

accuracy = (
    predicted_classes == y_test_cpu
).float().mean()

print(
    f"\nTest Accuracy: "
    f"{accuracy.item() * 100:.2f}%"
)


# ============================================================
# 18. Confusion Matrix
# ============================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test_cpu.numpy(),
        predicted_classes.numpy()
    )
)


# ============================================================
# 19. Classification Report
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test_cpu.numpy(),
        predicted_classes.numpy()
    )
)

torch.save(
    model.state_dict(),
    "churn_model.pth"
)


joblib.dump(
    scaler,
    "scaler.pkl"
)