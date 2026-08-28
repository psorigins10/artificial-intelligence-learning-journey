from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("../../04-data-science/data-bases/pokemon.csv")


X = df[
    [
        "HP",
        "Attack",
        "Defense",
        "Speed",
    ]
]

y = df["Legendary"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Scale the complete dataset only for visualization
X_scaled = scaler.transform(X)


knc = KNeighborsClassifier(
    n_neighbors=5
)

knc.fit(X_train_scaled, y_train)


y_pred = knc.predict(X_test_scaled)

print("Predicted:", y_pred)

accuracy = knc.score(X_test_scaled, y_test)

print(f"\nAccuracy: {accuracy}")

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

print(df["Legendary"].value_counts())

# Plot Attack vs Speed
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# HP vs Attack
axes[0, 0].scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=y
)
axes[0, 0].set_xlabel("HP (scaled)")
axes[0, 0].set_ylabel("Attack (scaled)")
axes[0, 0].set_title("HP vs Attack")


# HP vs Defense
axes[0, 1].scatter(
    X_scaled[:, 0],
    X_scaled[:, 2],
    c=y
)
axes[0, 1].set_xlabel("HP (scaled)")
axes[0, 1].set_ylabel("Defense (scaled)")
axes[0, 1].set_title("HP vs Defense")


# Attack vs Speed
axes[1, 0].scatter(
    X_scaled[:, 1],
    X_scaled[:, 3],
    c=y
)
axes[1, 0].set_xlabel("Attack (scaled)")
axes[1, 0].set_ylabel("Speed (scaled)")
axes[1, 0].set_title("Attack vs Speed")


# Defense vs Speed
axes[1, 1].scatter(
    X_scaled[:, 2],
    X_scaled[:, 3],
    c=y
)
axes[1, 1].set_xlabel("Defense (scaled)")
axes[1, 1].set_ylabel("Speed (scaled)")
axes[1, 1].set_title("Defense vs Speed")


plt.tight_layout()
plt.show()