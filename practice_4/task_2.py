import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets


def main():
    os.makedirs("results", exist_ok=True)

    data = datasets.load_digits(n_class=10)
    features = StandardScaler().fit_transform(data.data)
    th = 0.99

    print(f"Dataset shape: {features.shape}")

    pca = PCA(n_components=features.shape[1]).fit(features)
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    print("\nExplained variance ratio of the first 5 components:")
    print(pca.explained_variance_ratio_[:5])

    print("\nCumulative explained variance of the first 5 components:")
    print(cumulative[:5])

    n = np.searchsorted(cumulative, th).astype(int) + 1

    print(f"\nComponents needed for >= 99% explained variance: {n}")

    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(cumulative) + 1), cumulative)
    plt.axhline(y=th, color="red", linestyle="--", label="99% threshold")
    plt.axvline(x=n, color="green", linestyle="--", label=f"n={n}")
    plt.xlabel("Number of Components")
    plt.ylabel("Explained Variance")
    plt.title("Number of Components vs Explained Variance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/task_2_n_components_explained_variance.png")
    plt.show()


if __name__ == "__main__":
    main()
