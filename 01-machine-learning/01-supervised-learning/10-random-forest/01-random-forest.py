from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

df = pd.read_csv("../04-data-science/data-bases/pokemon.csv")

X = df[[
    "HP",
    "Attack",
    "Defense",
    "Sp. Atk",
    "Sp. Def",
    "Speed",
    "Generation"
]]

y = df["Legendary"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)

model = RandomForestClassifier(
    n_estimators = 100,
    random_state = 42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Predicted: {y_pred}")

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy Score: ", accuracy)
print(classification_report(y_test, y_pred))