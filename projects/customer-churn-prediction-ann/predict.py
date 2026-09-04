import torch
import torch.nn as nn
import pandas as pd
import joblib


# ============================================================
# 1. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")


# ============================================================
# 2. Define the same model
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
# 3. Load trained model
# ============================================================

model = ChurnANN()

model.load_state_dict(
    torch.load(
        "churn_model.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()


# ============================================================
# 4. Load scaler
# ============================================================

scaler = joblib.load("scaler.pkl")


# ============================================================
# 5. Get customer information
# ============================================================

print("\nEnter customer information:\n")

age = int(input("Age: "))
gender = input("Gender (Male/Female): ")
tenure = int(input("Tenure: "))
usage_frequency = int(input("Usage Frequency: "))
support_calls = int(input("Support Calls: "))
payment_delay = int(input("Payment Delay: "))

subscription_type = input(
    "Subscription Type (Basic/Premium/Standard): "
)

contract_length = input(
    "Contract Length (Annual/Monthly/Quarterly): "
)

total_spend = float(input("Total Spend: "))
last_interaction = int(input("Last Interaction: "))


# ============================================================
# 6. Create DataFrame
# ============================================================

customer = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Tenure": [tenure],
    "Usage Frequency": [usage_frequency],
    "Support Calls": [support_calls],
    "Payment Delay": [payment_delay],
    "Subscription Type": [subscription_type],
    "Contract Length": [contract_length],
    "Total Spend": [total_spend],
    "Last Interaction": [last_interaction]
})


# ============================================================
# 7. One-hot encode categorical features
# ============================================================

categorical_cols = [
    "Gender",
    "Subscription Type",
    "Contract Length"
]

customer = pd.get_dummies(
    customer,
    columns=categorical_cols,
    dtype=int
)


# ============================================================
# 8. Make sure we have exactly 15 features
# ============================================================

expected_columns = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",

    "Gender_Female",
    "Gender_Male",

    "Subscription Type_Basic",
    "Subscription Type_Premium",
    "Subscription Type_Standard",

    "Contract Length_Annual",
    "Contract Length_Monthly",
    "Contract Length_Quarterly"
]

customer = customer.reindex(
    columns=expected_columns,
    fill_value=0
)


# ============================================================
# 9. Scale the input
# ============================================================

customer_scaled = scaler.transform(customer)


# ============================================================
# 10. Convert to PyTorch tensor
# ============================================================

customer_tensor = torch.tensor(
    customer_scaled,
    dtype=torch.float32
).to(device)


# ============================================================
# 11. Make prediction
# ============================================================

with torch.no_grad():

    output = model(customer_tensor)

    probability = torch.sigmoid(output)

    prediction = (
        probability >= 0.5
    ).float()


# ============================================================
# 12. Display result
# ============================================================

probability = probability.item()
prediction = int(prediction.item())

print("\n==============================")
print("       CHURN PREDICTION")
print("==============================")

print(
    f"Churn Probability: {probability * 100:.2f}%"
)

if prediction == 1:
    print("Prediction: 🚨 CUSTOMER WILL CHURN")
else:
    print("Prediction: ✅ CUSTOMER WILL NOT CHURN")