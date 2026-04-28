from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from collections import Counter
import numpy as np


class KNN:
    def __init__(self, k, metric='L2'):
        self.X_train = None
        self.y_train = None
        self.k = k
        self.metric = metric.upper()
        if self.metric not in ['L1', 'L2']:
            raise ValueError("metric must be 'L1' or 'L2'")

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        if self.metric == "L2":
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        else:
            distances = np.sum(np.abs(self.X_train - x), axis=1)

        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        return Counter(k_labels).most_common(1)[0][0]


def evaluate_knn_model(k, metric, X_train, y_train, X_test, y_test, target_names):
    print(f"KNN with {metric} distance:")
    model = KNN(k=k, metric=metric)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy = {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    return accuracy


def main():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    acc_l2 = evaluate_knn_model(k=2, metric='L2', X_train=X_train, y_train=y_train,
                                X_test=X_test, y_test=y_test, target_names=iris.target_names)

    acc_l1 = evaluate_knn_model(k=3, metric='L1', X_train=X_train, y_train=y_train,
                                X_test=X_test, y_test=y_test, target_names=iris.target_names)

    print("Comparison:")
    print(f"L2 Accuracy: {acc_l2 * 100:.2f}%")
    print(f"L1 Accuracy: {acc_l1 * 100:.2f}%")


if __name__ == '__main__':
    main()