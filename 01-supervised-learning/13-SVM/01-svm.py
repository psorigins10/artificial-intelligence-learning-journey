from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X, y = make_classification(
            n_features = 5, 
            random_state = 0,
            n_samples = 1000
            )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify = y
)

clf = make_pipeline(
        StandardScaler(),
        LinearSVC(random_state=0, tol=1e-5)
        )

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(y_pred)

print(f"\n{classification_report(y_test, y_pred)}")