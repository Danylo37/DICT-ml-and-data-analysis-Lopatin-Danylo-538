from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
import os

RESULTS_DIR = "results"


class KNNImputer:
    def __init__(self, k=3, metric='L2'):
        self.k = k
        self.metric = metric.upper()

    def fit(self, X):
        self.X_train = np.array(X, dtype=float)
        return self

    def _distance(self, x1, x2):
        mask = ~np.isnan(x1) & ~np.isnan(x2)

        if np.sum(mask) == 0:
            return np.inf

        if self.metric == "L2":
            return np.sqrt(np.sum((x1[mask] - x2[mask]) ** 2))
        else:
            return np.sum(np.abs(x1[mask] - x2[mask]))

    def transform(self, X):
        X = np.array(X, dtype=float).copy()
        eps = 1e-8

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):

                if np.isnan(X[i, j]):
                    candidates = []

                    for t in range(self.X_train.shape[0]):
                        if not np.isnan(self.X_train[t, j]):
                            dist = self._distance(X[i], self.X_train[t])
                            if np.isfinite(dist):
                                candidates.append((dist, self.X_train[t, j]))

                    if candidates:
                        candidates.sort(key=lambda x: x[0])
                        neighbors = candidates[:self.k]

                        distances = np.array([d for d, _ in neighbors], dtype=float)
                        values = np.array([v for _, v in neighbors], dtype=float)

                        weights = 1.0 / (distances + eps)
                        X[i, j] = np.sum(weights * values) / np.sum(weights)
                    else:
                        col_mean = np.nanmean(self.X_train[:, j])
                        X[i, j] = col_mean if not np.isnan(col_mean) else 0.0

        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def introduce_missing_values(X, missing_rate):
    X_missing = X.copy()
    n_samples, n_features = X.shape
    n_missing = int(n_samples * n_features * missing_rate)

    indices = np.random.choice(n_samples * n_features, n_missing, replace=False)

    for idx in indices:
        i = idx // n_features
        j = idx % n_features
        X_missing[i, j] = np.nan

    return X_missing


def run_experiment(metric, X_train_missing, X_test, y_train, y_test):
    imputer = KNNImputer(k=3, metric=metric)
    X_train_imputed = imputer.fit_transform(X_train_missing)
    X_test_imputed = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)

    return accuracy_score(y_test, y_pred)


def plot_missing_heatmap(X, title, filename):
    mask = np.isnan(X)

    plt.figure()
    plt.imshow(mask, aspect="auto")
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Samples")
    plt.colorbar(label="Missing (1=True)")
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.show()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.random.seed(42)

    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    for missing_rate in [0.1, 0.3, 0.5, 0.7, 0.9]:
        print(f"Missing rate: {missing_rate}:")

        X_train_missing = introduce_missing_values(X_train, missing_rate=missing_rate)

        plot_missing_heatmap(
            X_train_missing,
            f"Before imputation (rate={missing_rate})",
            f"heatmap_before_{missing_rate}.png"
        )

        acc_l2 = run_experiment('L2', X_train_missing, X_test, y_train, y_test)
        acc_l1 = run_experiment('L1', X_train_missing, X_test, y_train, y_test)

        print(f"L2 accuracy: {acc_l2 * 100:.2f}%")
        print(f"L1 accuracy: {acc_l1 * 100:.2f}%")
        print()


if __name__ == "__main__":
    main()
