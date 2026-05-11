from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import datasets
import os


RESULTS_DIR = "results"
K = 10


def main():
    data = datasets.load_digits(n_class=K)

    features = StandardScaler().fit_transform(data.data)
    labels = data.target

    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)

    tsne = TSNE(n_components=2, n_iter_without_progress=150, random_state=42)
    features_tsne = tsne.fit_transform(features)

    kmeans_pca = KMeans(n_clusters=K, random_state=42, n_init='auto').fit(features_pca)
    kmeans_tsne = KMeans(n_clusters=K, random_state=42, n_init='auto').fit(features_tsne)

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(2, 2, 1)
    scatter = ax.scatter(features_pca[:, 0], features_pca[:, 1], c=labels)
    ax.legend(scatter.legend_elements()[0], labels, loc="lower right", title="Classes")
    ax.set_title("PCA")

    ax = fig.add_subplot(2, 2, 2)
    scatter = ax.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels)
    ax.legend(scatter.legend_elements()[0], labels, loc="lower right", title="Classes")
    ax.set_title("tSNE")

    ax = fig.add_subplot(2, 2, 3)
    scatter = ax.scatter(features_pca[:, 0], features_pca[:, 1], c=kmeans_pca.labels_)
    ax.legend(scatter.legend_elements()[0], labels, loc="lower right", title="Clusters")
    ax.set_title("Clustered PCA")

    ax = fig.add_subplot(2, 2, 4)
    scatter = ax.scatter(features_tsne[:, 0], features_tsne[:, 1], c=kmeans_tsne.labels_)
    ax.legend(scatter.legend_elements()[0], labels, loc="lower right", title="Clusters")
    ax.set_title("Clustered tSNE")

    plt.savefig(os.path.join(RESULTS_DIR, "task_2_pca_vs_tsne.png"))
    plt.show()


if __name__ == '__main__':
    main()