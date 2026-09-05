import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from model import MLP

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using Device: {device}")
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
test_loader = DataLoader(test_data, batch_size = 64)

model = MLP().to(device)

lossFunction = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
epochs = 20

best_accuracy = 0.0

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        prediction = model(images)

        loss = lossFunction(prediction, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

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

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        torch.save(
            model.state_dict(),
            "best_mnist_mlp.pth"
        )

        print("✓ Best model saved!")

print(f"\nBest accuracy: {best_accuracy * 100:.2f}%")
print("Best model saved as: best_mnist_mlp.pth")