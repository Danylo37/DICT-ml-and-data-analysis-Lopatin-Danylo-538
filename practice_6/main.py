import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import umap


def draw_umap(X, y, n_neighbors=15, min_dist=0.1, metric='euclidean'):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric=metric,
        low_memory=False
    )
    features_umap = reducer.fit_transform(X)

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        features_umap[:, 0],
        features_umap[:, 1],
        c=y,
        cmap="tab10",
        s=10
    )

    plt.colorbar(scatter)
    plt.title(f"UMAP visualization of Fashion-MNIST embeddings\n(n_neighbors = {n_neighbors}, min_dist = {min_dist})")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")

    plt.savefig(f"results/UMAP_embeddings_{n_neighbors}_neighbors_{min_dist}_dist.png")
    plt.show()


def main():
    os.makedirs("results", exist_ok=True)

    (X_train, y_train), (_, _) = fashion_mnist.load_data()

    X = X_train.reshape(X_train.shape[0], -1)


    for n in (2, 5, 15, 25, 50, 100, 200):
        draw_umap(X, y_train, n_neighbors=n)

    for n in (0.2, 0.4, 0.6, 0.8, 1.0):
        draw_umap(X, y_train, min_dist=n)


if __name__ == "__main__":
    main()