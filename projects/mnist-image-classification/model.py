from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

train_data = datasets.MNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor()
)

test_data = datasets.MNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
)

train_loader = DataLoader(train_data, batch_size = 64, shuffle = True)
test_loader  = DataLoader(test_data, batch_size = 64)

class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)

model = MLP().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
epochs = 15

best_accuracy = 0.0

for epoch in range(epochs):

    # ---------- TRAIN ----------
    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        prediction = model(images)

        loss = loss_fn(prediction, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    # ---------- VALIDATE ----------
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)

            predicted_labels = predictions.argmax(dim=1)

            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()

    accuracy = correct / total

    print(
        f"Epoch {epoch + 1} | "
        f"Loss: {average_loss:.4f} | "
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    # ---------- SAVE BEST ----------
    if accuracy > best_accuracy:

        best_accuracy = accuracy

        torch.save(
            model.state_dict(),
            "best_mnist_mlp.pth"
        )

        print("  ✓ Best model saved!")
        
print(f"\nBest accuracy: {best_accuracy * 100:.2f}%")
print("Best model saved as: best_mnist_mlp.pth")