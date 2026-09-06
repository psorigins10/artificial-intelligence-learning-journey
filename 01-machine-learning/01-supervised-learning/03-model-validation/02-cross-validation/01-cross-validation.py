from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.svm import SVC

X, y = make_classification(
    n_samples = 1000,
    n_features = 5,
    random_state = 0
)

clf = SVC(
    kernel = "linear"
)

# Seperates the data into folds contoled my 'cv'
score = cross_val_score(
    clf,
    X,
    y,
    cv = 10
)

print(score)
print(score.mean())
print("Standard deviation:", score.std())