import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class MyPLS:
    def __init__(self, n_components):
        self.n_components = n_components

        self.W = None
        self.P = None
        self.q = None

        self.B = None
        self.B0 = None

        self.x_mean_ = None
        self.y_mean_ = None
        
        self.x_std_ = None
        self.y_std_ = None


    def fit(self, X, y):
        self.x_mean_ = np.mean(X, axis=0)
        self.y_mean_ = np.mean(y)

        self.x_std_ = np.std(X, axis=0)
        self.y_std_ = np.std(y)

        X = (X - self.x_mean_) / self.x_std_
        y = ((y - self.y_mean_) / self.y_std_).reshape(-1, 1)

        Xk = X.copy()

        W = []
        P = []
        q = []

        w = Xk.T @ y
        w = w / np.linalg.norm(w)

        for k in range(self.n_components):

            t = Xk @ w
            tk = (t.T @ t)[0, 0]

            p = Xk.T @ t / tk
            qk = (y.T @ t)[0, 0] / tk

            W.append(w.flatten())
            P.append(p.flatten())
            q.append(qk)

            if qk == 0:
                break

            if k < self.n_components - 1:
                Xk = Xk - t @ p.T
                w = Xk.T @ y
                w = w / np.linalg.norm(w)

        self.W = np.column_stack(W)
        self.P = np.column_stack(P)
        self.q = np.array(q).reshape(-1, 1)

        B = self.W @ np.linalg.inv(self.P.T @ self.W) @ self.q
        B0 = self.q[0] - self.P[:, 0].T @ B

        self.B = B.flatten()
        self.B0 = B0

        return self


    def predict(self, X):
        Xc = (X - self.x_mean_) / self.x_std_
        y_pred = Xc @ self.B + self.B0
        return y_pred * self.y_std_ + self.y_mean_


def load_data():
    data = pd.read_csv("WineQT.csv")

    X = data.drop("quality", axis=1).values.astype(np.float32)
    y = data["quality"].values.astype(np.float32)

    return X, y


def main():
    os.makedirs("results", exist_ok=True)

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    n_components = 5

    pls_my = MyPLS(n_components)
    pls_my.fit(X_train, y_train)

    y_pred_my = pls_my.predict(X_test)

    pls_sk = PLSRegression(n_components=n_components)
    pls_sk.fit(X_train, y_train)

    y_pred_sk = pls_sk.predict(X_test).ravel()

    mse_my = mean_squared_error(y_test, y_pred_my)
    mse_sk = mean_squared_error(y_test, y_pred_sk)
    
    r2_my = r2_score(y_test, y_pred_my)
    r2_sk = r2_score(y_test, y_pred_sk)

    print("My PLS MSE:", mse_my)
    print("Sklearn PLS MSE:", mse_sk)

    print("\nMy PLS R2:", r2_my)
    print("Sklearn PLS R2:", r2_sk)

    print("\nMy PLS coefficients:")
    print(pls_my.B)

    print("\nSklearn PLS coefficients:")
    print(pls_sk.coef_.ravel())

    min_val = min(y_test.min(), y_pred_my.min(), y_pred_sk.min())
    max_val = max(y_test.max(), y_pred_my.max(), y_pred_sk.max())

    plt.scatter(y_test, y_pred_my, label="My PLS", alpha=0.7)
    plt.plot([min_val, max_val], [min_val, max_val], label="Perfect prediction", color='r')
    plt.xlabel("True Quality")
    plt.ylabel("Predicted Quality")
    plt.title("My PLS Prediction")
    plt.legend()
    plt.savefig("results/pls_prediction_comparison_my.png")
    plt.show()

    plt.scatter(y_test, y_pred_sk, label="Sklearn PLS", alpha=0.7)
    plt.plot([min_val, max_val], [min_val, max_val], label="Perfect prediction", color='r')
    plt.xlabel("True Quality")
    plt.ylabel("Predicted Quality")
    plt.title("Sklearn PLS Prediction")
    plt.legend()
    plt.savefig("results/pls_prediction_comparison_sklearn.png")
    plt.show()


if __name__ == "__main__":
    main()


