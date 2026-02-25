from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np


def plot_2d_scatter(features, labels, title, filename, dpi=300):
    _, ax = plt.subplots()
    ax.scatter(features[:, 0], features[:, 1], c=labels)
    ax.set_title(title)
    plt.savefig(filename, dpi=dpi)
    plt.show()
    plt.close()


def test_single_cluster_std(cluster_std_value):
    features, labels = make_blobs(n_samples=1000,
                                  n_features=2,
                                  centers=3,
                                  cluster_std=cluster_std_value,
                                  shuffle=True,
                                  random_state=42)

    plot_2d_scatter(features, labels,
                    f'Original clusters (cluster_std={cluster_std_value})',
                    f'results/task_2_original_clusters_cluster_std_{cluster_std_value}.png')

    model = KMeans(n_clusters=3, random_state=42).fit(features)
    predicted = model.labels_

    score = silhouette_score(features, predicted)
    print(f'Cluster_std: {cluster_std_value}, Silhouette Score: {score:.2f}')

    plot_2d_scatter(features, predicted,
                    f'KMeans predictions (cluster_std={cluster_std_value})',
                    f'results/task_2_kmeans_predictions_cluster_std_{cluster_std_value}.png')

    return score


def compare_cluster_std_values():
    cluster_std_values = np.linspace(0.1, 10, 20)
    silhouette_scores = []

    for std_value in cluster_std_values:
        features, labels = make_blobs(n_samples=1000,
                                      n_features=2,
                                      centers=3,
                                      cluster_std=std_value,
                                      shuffle=True,
                                      random_state=42)

        model = KMeans(n_clusters=3, random_state=42).fit(features)
        predicted = model.labels_
        score = silhouette_score(features, predicted)
        silhouette_scores.append(score)
        print(f'cluster_std={std_value:.2f}, silhouette_score={score:.2f}')

    plt.figure(figsize=(10, 6))
    plt.plot(cluster_std_values, silhouette_scores, marker='o', linewidth=2, markersize=6)
    plt.xlabel('cluster_std', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Dependency of Silhouette Score on cluster_std', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/task_2_silhouette_score_vs_cluster_std.png', dpi=300)
    plt.show()
    plt.close()


def compare_centers_values():
    centers_values = range(2, 21)  # From 2 to 20 centers
    silhouette_scores = []

    for n_centers in centers_values:
        features, labels = make_blobs(n_samples=1000,
                                      n_features=2,
                                      centers=n_centers,
                                      cluster_std=1.0,
                                      shuffle=True,
                                      random_state=42)

        model = KMeans(n_clusters=n_centers, random_state=42).fit(features)
        predicted = model.labels_
        score = silhouette_score(features, predicted)
        silhouette_scores.append(score)
        print(f'centers={n_centers}, silhouette_score={score:.2f}')

    plt.figure(figsize=(10, 6))
    plt.plot(list(centers_values), silhouette_scores, marker='o', linewidth=2, markersize=6)
    plt.xlabel('Number of Centers', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Dependency of Silhouette Score on Number of Centers', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/task_2_silhouette_score_vs_centers.png', dpi=300)
    plt.show()
    plt.close()


def main():
    print("Testing single cluster_std value:")
    test_single_cluster_std(37)

    print("\nComparing different cluster_std values:")
    compare_cluster_std_values()

    print("\nComparing different numbers of centers:")
    compare_centers_values()


if __name__ == "__main__":
    main()
