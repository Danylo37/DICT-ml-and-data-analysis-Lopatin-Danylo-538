from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import sklearn.datasets
import numpy as np
import umap.plot
import umap
import os


def mnist_digits():
    data, labels = sklearn.datasets.fetch_openml("mnist_784", version=1, return_X_y=True)

    mapper = umap.UMAP(random_state=42).fit(data)

    umap.plot.points(mapper, labels=labels)
    plt.title("MNIST digits dataset")
    plt.tight_layout()
    plt.savefig("results/mnist_digits.png")
    plt.show()

    corners = np.array([
        [-5, -10],
        [-7, 6],
        [2, -8],
        [12, 4],
    ])

    test_pts = np.array([
        (corners[0] * (1 - x) + corners[1] * x) * (1 - y) +
        (corners[2] * (1 - x) + corners[3] * x) * y
        for y in np.linspace(0, 1, 10)
        for x in np.linspace(0, 1, 10)
    ])

    inv_transformed_points = mapper.inverse_transform(test_pts)

    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(10, 20, fig)
    scatter_ax = fig.add_subplot(gs[:, :10])
    digit_axes = np.zeros((10, 10), dtype=object)
    for i in range(10):
        for j in range(10):
            digit_axes[i, j] = fig.add_subplot(gs[i, 10 + j])

    scatter_ax.scatter(mapper.embedding_[:, 0], mapper.embedding_[:, 1],
                       c=labels.astype(np.int32), cmap="Spectral", s=0.1)
    scatter_ax.set(xticks=[], yticks=[])
    scatter_ax.scatter(test_pts[:, 0], test_pts[:, 1], marker='x', c='k', s=15)

    for i in range(10):
        for j in range(10):
            digit_axes[i, j].imshow(inv_transformed_points[i * 10 + j].reshape(28, 28))
            digit_axes[i, j].set(xticks=[], yticks=[])

    plt.title("UMAP embedding of MNIST digits with inverse transformed points")
    plt.tight_layout()
    plt.savefig("results/mnist_digits_inverse.png")
    plt.show()


def fashion_mnist():
    data, labels = sklearn.datasets.fetch_openml('Fashion-MNIST', version=1, return_X_y=True)

    mapper = umap.UMAP(random_state=42).fit(data)

    umap.plot.points(mapper, labels=labels)
    plt.title("Fashion MNIST dataset")
    plt.tight_layout()
    plt.savefig("results/fashion_mnist.png")
    plt.show()

    corners = np.array([
        [-2, -6],
        [-9, 3],
        [7, -5],
        [4, 10],
    ])

    test_pts = np.array([
        (corners[0] * (1 - x) + corners[1] * x) * (1 - y) +
        (corners[2] * (1 - x) + corners[3] * x) * y
        for y in np.linspace(0, 1, 10)
        for x in np.linspace(0, 1, 10)
    ])

    inv_transformed_points = mapper.inverse_transform(test_pts)

    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(10, 20, fig)
    scatter_ax = fig.add_subplot(gs[:, :10])
    digit_axes = np.zeros((10, 10), dtype=object)
    for i in range(10):
        for j in range(10):
            digit_axes[i, j] = fig.add_subplot(gs[i, 10 + j])

    scatter_ax.scatter(mapper.embedding_[:, 0], mapper.embedding_[:, 1],
                       c=labels.astype(np.int32), cmap='Spectral', s=0.1)
    scatter_ax.set(xticks=[], yticks=[])
    scatter_ax.scatter(test_pts[:, 0], test_pts[:, 1], marker='x', c='k', s=15)

    for i in range(10):
        for j in range(10):
            digit_axes[i, j].imshow(inv_transformed_points[i * 10 + j].reshape(28, 28))
            digit_axes[i, j].set(xticks=[], yticks=[])

    plt.title("UMAP embedding of Fashion MNIST with inverse transformed points")
    plt.tight_layout()
    plt.savefig("results/fashion_mnist_inverse.png")
    plt.show()


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    mnist_digits()
    fashion_mnist()
