from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import datasets
import os


RESULTS_DIR = "results"
MAX_CLUSTERS = 30


def plot_clusters_comparison(features, iris, n_clusters, model, results_dir):
    print(f"\nPredicted clusters for {n_clusters} clusters:")
    print(model.labels_)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(features[:, 0], features[:, 1], c=iris.target)
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.title("True Classes")

    plt.subplot(1, 2, 2)
    plt.scatter(features[:, 0], features[:, 1], c=model.labels_)
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.title(f"KMeans Clusters ({n_clusters})")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"task_1_iris_kmeans_{n_clusters}_clusters.png"))
    plt.show()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    iris = datasets.load_iris()
    features = iris.data

    scaler = StandardScaler()
    features_std = scaler.fit_transform(features)

    plt.figure(figsize=(8, 5))

    scatter = plt.scatter(features[:, 0], features[:, 1], c=iris.target)

    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.title("Iris Dataset - True Classes")

    plt.legend(
        scatter.legend_elements()[0],
        iris.target_names,
        loc="lower right",
        title="Classes",
    )

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "task_1_iris_true_classes.png"))
    plt.show()

    kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init="auto")

    model_3 = kmeans_3.fit(features_std)
    plot_clusters_comparison(features, iris, 3, model_3, RESULTS_DIR)

    inertias = []

    for n_clusters in range(1, MAX_CLUSTERS + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")

        kmeans.fit(features_std)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))

    plt.plot(range(1, MAX_CLUSTERS + 1), inertias, marker="o")

    plt.title("Elbow Method for KMeans")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "task_1_iris_elbow_method.png"))
    plt.show()

    kmeans_6 = KMeans(n_clusters=6, random_state=42, n_init="auto")

    model_6 = kmeans_6.fit(features_std)
    plot_clusters_comparison(features, iris, 6, model_6, RESULTS_DIR)


if __name__ == "__main__":
    main()
