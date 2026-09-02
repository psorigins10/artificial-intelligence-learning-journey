import numpy as np

class LinearRegressionGD:

    def __init__(self):
        self.coef_ = 0
        self.intercept_ = 0

    def fit(self, X_train, y_train):
        learningRate =  0.001

        while True:

        # For Coef
            loss_slop = -2 * np.mean(np.ravel(X_train) * (y_train - (self.coef_ * np.ravel(X_train) + self.intercept_)))
            m_step_size = loss_slop * learningRate
            m_new = self.coef_ - m_step_size

        # For Intercept
            loss_slop = -2 * np.mean(y_train - (self.coef_ * np.ravel(X_train) + self.intercept_))
            b_step_size = loss_slop * learningRate
            b_new = self.intercept_ - b_step_size

            if abs(b_new - self.intercept_) < 0.000001 and abs(m_new - self.coef_) < 0.000001:
                break
            else:
                self.intercept_ = b_new
                self.coef_ = m_new


    def predict(self, X_test):

        y_pred = self.coef_ * np.ravel(X_test) + self.intercept_
        return y_pred