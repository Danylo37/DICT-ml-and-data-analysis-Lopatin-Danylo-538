from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "results"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    k = 5

    iris = load_iris()
    X, y = iris.data, iris.target

    standardizer = StandardScaler()
    X_std = standardizer.fit_transform(X)

    cmap = plt.cm.Set3
    plt.title("Iris Dataset")
    scatter = plt.scatter(X[:, 0], X[:, 1], c=iris.target, cmap=cmap)
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.legend(scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes")
    plt.savefig(os.path.join(RESULTS_DIR, "task_1_iris_scatter.png"))
    plt.show()

    nearest_neighbors = NearestNeighbors(n_neighbors=k).fit(X_std)

    new_observation = [1, 1, 1, 1]
    new_observation_standardized = standardizer.transform([new_observation])

    distances, indices = nearest_neighbors.kneighbors(new_observation_standardized)

    print(f"Distances: {distances}")
    print(f"Indices: {indices}")

    print("\nStandardized feature values of the 5 nearest neighbors:")
    print(X_std[indices])

    knn = KNeighborsClassifier(n_neighbors=k).fit(X_std, y)

    new_observations = [[5, 4, 2, 1], [7, 3, 3, 4]]
    new_observations_std = standardizer.transform(new_observations)

    y_pred = knn.predict(new_observations_std)

    print(f"\nNew observations:")
    for obs, pred in zip(new_observations, y_pred):
        print(f"{obs} -> {iris.target_names[pred]}")

    pipe = Pipeline([("standardizer", standardizer), ("knn", knn)])

    search_space = [{"knn__n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]

    classifier = GridSearchCV(pipe, search_space, cv=5, verbose=0).fit(X, y)

    best_k = classifier.best_estimator_.get_params()["knn__n_neighbors"]
    print(f"\nBest k: {best_k}")

    print(f"\nAccuracy with k={k}: {knn.score(X_std, y) * 100:.2f}%")
    knn = KNeighborsClassifier(n_neighbors=best_k).fit(X_std, y)
    print(f"Accuracy with k={best_k}: {knn.score(X_std, y) * 100:.2f}%")

    plt.title("Iris Dataset with New Observations")
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap)
    plt.scatter(
        [obs[0] for obs in new_observations],
        [obs[1] for obs in new_observations],
        c=y_pred,
        cmap=cmap,
        marker="X",
        s=200,
        edgecolors="black",
        label="New observations"
    )
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.legend(scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes")
    plt.savefig(os.path.join(RESULTS_DIR, "task_1_iris_with_new_points.png"))
    plt.show()


if __name__ == "__main__":
    main()