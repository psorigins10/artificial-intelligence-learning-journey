import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ---------------- DEVICE ----------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# ---------------- MODEL ----------------

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


# ---------------- LOAD MODEL ----------------

model = MLP().to(device)

model.load_state_dict(
    torch.load(
        "best_mnist_mlp.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Model loaded!")


# ---------------- IMAGE PREPROCESSING ----------------

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])


# ---------------- LOAD IMAGE ----------------

image = Image.open("digit8.png")

image = transform(image)

# Add batch dimension
image = image.unsqueeze(0)
image = image.to(device)


print("Image shape:", image.shape)


# ---------------- PREDICTION ----------------

with torch.no_grad():

    output = model(image)

    prediction = output.argmax(dim=1).item()


print(f"Model prediction: {prediction}")