from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


def plot_3d_scatter(features, labels, title, filename):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=labels, cmap='viridis', s=50)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()


def main():
    features, labels = make_blobs(n_samples=1000,
                                  n_features=3,
                                  centers=3,
                                  cluster_std=0.37,
                                  shuffle=True,
                                  random_state=42)

    plot_3d_scatter(features, labels, 'Original Clusters', 'results/task_3_original_clusters.png')

    model = KMeans(n_clusters=3, random_state=42).fit(features)

    predicted = model.labels_

    score = silhouette_score(features, predicted)
    print(f'Silhouette Score: {score:.2f}')

    plot_3d_scatter(features, predicted, 'KMeans Predictions', 'results/task_3_kmeans_predictions.png')


if __name__ == '__main__':
    main()