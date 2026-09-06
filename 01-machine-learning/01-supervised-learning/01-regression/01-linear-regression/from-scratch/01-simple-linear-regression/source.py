class LinearRegression:
    def __init__(self):
        self.coff_ = None
        self.intercept = None

    def fit(self, X_train, y_train):

        num = 0
        den = 0

        for i in range(X_train.shape[0]):
            num += ((X_train[i] - X_train.mean()) * (y_train[i] - y_train.mean()))
            den += (X_train[i] - X_train.mean()) ** 2

        self.coff_ = num / den
        self.intercept = y_train.mean() - (self.coff_ * X_train.mean())

    def predict(self, X_test):

        y_pred = (self.coff_ * X_test) + self.intercept
        return [y_pred, self.coff_, self.intercept]