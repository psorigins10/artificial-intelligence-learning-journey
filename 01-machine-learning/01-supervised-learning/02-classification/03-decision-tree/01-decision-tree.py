import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

df = pd.read_csv("../../04-data-science/data-bases/pokemon.csv")

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

clf = tree.DecisionTreeClassifier(
    class_weight ="balanced",
    random_state=42
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Predicted: {y_pred}")

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy Score: ", accuracy)
print(classification_report(y_test, y_pred))

plt.figure(figsize=(20, 12))

tree.plot_tree(
    clf,
    feature_names=X.columns,
    class_names=["Not Legendary", "Legendary"],
    filled=True
)

plt.show()