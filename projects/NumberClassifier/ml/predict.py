import torch
import torch.nn as nn
from torchvision import transforms
from .model import MLP

import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_mnist_mlp.pth")

model = MLP().to(device)
state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True,
)
model.load_state_dict(state_dict)
model.eval()
print("Model Loaded Successfully!")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])


def predict(img):
    image = transform(img)
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1).item()

    return prediction