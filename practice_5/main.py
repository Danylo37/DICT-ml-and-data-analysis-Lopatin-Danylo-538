import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.datasets import fashion_mnist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def main():
    (X_train, y_train), (_, _) = fashion_mnist.load_data()

    print("Before reshaping:")
    print(X_train.shape)

    X = X_train.reshape(X_train.shape[0], -1)

    print("\nAfter reshaping:")
    print(X.shape)

    n_samples = 3000

    X_subset = X[:n_samples]
    y_subset = y_train[:n_samples]

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        random_state=42,
    )

    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_subset)

    X_embedded = tsne.fit_transform(X_pca)

    print("\nAfter t-SNE transformation:")
    print(X_embedded.shape)

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        X_embedded[:, 0],
        X_embedded[:, 1],
        c=y_subset,
        cmap="tab10",
        s=10
    )

    plt.colorbar(scatter)
    plt.title("t-SNE visualization of Fashion-MNIST embeddings")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    plt.savefig("results/t-SNE_embeddings.png")
    plt.show()


if __name__ == '__main__':
    main()