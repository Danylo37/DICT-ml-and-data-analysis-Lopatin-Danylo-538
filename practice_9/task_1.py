from utils import load_dataset, preprocess, show_and_save_plot, plot_accuracy_comparison, predict_image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import os


class CustomLogRegression:
    def __init__(self, learning_rate=0.005, num_iterations=2000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.w = None
        self.b = 0.0
        self.costs = []

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _initialize(self, dim):
        self.w = np.zeros((dim, 1))
        self.b = 0.0

    def _propagate(self, X, Y):
        m = X.shape[1]
        A = self._sigmoid(np.dot(self.w.T, X) + self.b)
        eps = 1e-15
        cost = -np.sum(Y * np.log(A + eps) + (1 - Y) * np.log(1 - A + eps)) / m
        dw = np.dot(X, (A - Y).T) / m
        db = np.sum(A - Y) / m
        return dw, db, cost

    def fit(self, X, Y):
        self._initialize(X.shape[0])
        self.costs = []

        for i in range(self.num_iterations):
            dw, db, cost = self._propagate(X, Y)
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            if i % 100 == 0:
                self.costs.append(cost)

        return self

    def predict_proba(self, X):
        return self._sigmoid(np.dot(self.w.T, X) + self.b)

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs > 0.5).astype(int)

    def score(self, X, Y):
        preds = self.predict(X)
        return 100.0 * (1.0 - np.mean(np.abs(preds - Y)))


def main():
    os.makedirs("results", exist_ok=True)

    X_train, y_train, X_test, y_test, classes = load_dataset()
    X_train, X_test = preprocess(X_train, X_test)

    custom_model = CustomLogRegression(learning_rate=0.005, num_iterations=2000)
    custom_model.fit(X_train, y_train)

    custom_train_acc = custom_model.score(X_train, y_train)
    custom_test_acc = custom_model.score(X_test, y_test)

    print(f"Custom model train accuracy: {custom_train_acc:.2f}%")
    print(f"Custom model test accuracy: {custom_test_acc:.2f}%")

    sklearn_model = LogisticRegression(max_iter=2000)
    sklearn_model.fit(X_train.T, y_train.ravel())

    sk_train_pred = sklearn_model.predict(X_train.T)
    sk_test_pred = sklearn_model.predict(X_test.T)

    sk_train_acc = accuracy_score(y_train.ravel(), sk_train_pred) * 100.0
    sk_test_acc = accuracy_score(y_test.ravel(), sk_test_pred) * 100.0

    print(f"\nSklearn model train accuracy: {sk_train_acc:.2f}%")
    print(f"Sklearn model test accuracy: {sk_test_acc:.2f}%\n")

    show_and_save_plot(
        title="custom_cost_curve",
        x=np.arange(len(custom_model.costs)) * 100,
        y=custom_model.costs,
        xlabel="Iterations",
        ylabel="Cost",
    )

    plot_accuracy_comparison(custom_train_acc, custom_test_acc, sk_train_acc, sk_test_acc)

    # Predict on custom image
    predict_image(custom_model, classes, "./input/non-cat.png")


if __name__ == "__main__":
    main()
