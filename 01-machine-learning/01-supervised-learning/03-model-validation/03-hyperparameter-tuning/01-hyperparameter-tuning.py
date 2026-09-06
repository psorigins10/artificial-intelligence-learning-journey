from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import make_classification
clf = SVC()

X, y = make_classification(
    n_samples = 100,
    n_features = 2,
    n_redundant = 0,
    n_informative = 2,
    random_state = 42
)

# Specifing hyperparameters to search over
param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}

# Finds the best hyperparameters for the model
grid = GridSearchCV(
    clf,
    param_grid,
    cv=5
)

grid.fit(X, y)

print(grid.best_params_)
print(grid.best_score_)