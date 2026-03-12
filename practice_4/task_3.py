import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.datasets import fashion_mnist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


class MyPCA:
    def __init__(self, n_components):
        self.n_components = n_components

        self.explained_variance_ratio_ = None
        self.explained_variance_ = None
        self.components_ = None
        self.mean_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)

        X_centered = X - self.mean_

        cov = (X_centered.T @ X_centered) / X_centered.shape[0]

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.components_ = eigenvectors[:, :self.n_components]

        self.explained_variance_ = eigenvalues[:self.n_components]

        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

        return self

    def transform(self, X):
        X_centered = X - self.mean_
        return X_centered @ self.components_

    def inverse_transform(self, X_reduced):
        return X_reduced @ self.components_.T + self.mean_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def main():
    (x_train, y_train), _ = fashion_mnist.load_data()

    X = x_train.reshape(-1, 784).astype(np.float32)

    n_components = 50

    pca_my = MyPCA(n_components)
    X_reduced_my = pca_my.fit_transform(X)
    X_rec_my = pca_my.inverse_transform(X_reduced_my)

    pca_sk = PCA(n_components=n_components)
    X_reduced_sk = pca_sk.fit_transform(X)
    X_rec_sk = pca_sk.inverse_transform(X_reduced_sk)

    mse_my = np.mean((X - X_rec_my) ** 2)
    mse_sk = np.mean((X - X_rec_sk) ** 2)

    print("My PCA MSE:", mse_my)
    print("Sklearn PCA MSE:", mse_sk)

    print("\nMy PCA explained variance ratio (first 5):")
    print(pca_my.explained_variance_ratio_[:5])

    print("\nSklearn PCA explained variance ratio (first 5):")
    print(pca_sk.explained_variance_ratio_[:5])

    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(X[0].reshape(28, 28), cmap="gray")

    plt.subplot(1, 3, 2)
    plt.title("Reconstructed (My)")
    plt.imshow(X_rec_my[0].reshape(28, 28), cmap="gray")

    plt.subplot(1, 3, 3)
    plt.title("Reconstructed (Sklearn)")
    plt.imshow(X_rec_sk[0].reshape(28, 28), cmap="gray")

    plt.savefig("results/task_3_reconstruction_comparison_pca.png")
    plt.show()


if __name__ == "__main__":
    main()