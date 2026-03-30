from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from sklearn import svm
import os


def generate_data(n_samples_1=1000, n_samples_2=100):
    centers = [[0.0, 0.0], [2.0, 2.0]]
    clusters_std = [1.5, 0.5]

    X, y = make_blobs(
        n_samples=[n_samples_1, n_samples_2],
        centers=centers,
        cluster_std=clusters_std,
        random_state=0,
        shuffle=False,
    )
    return X, y


def train_model(X_train, y_train, class_weight=None):
    model = svm.SVC(kernel="linear", class_weight=class_weight)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, title):
    y_pred = model.predict(X_test)

    print(f"Classification Report ({title}):")
    print(classification_report(y_test, y_pred, zero_division=0))
    print()


def plot_results(X, y, models, title):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors="k")

    ax = plt.gca()
    colors = ["k", "r", "b"]
    labels = []

    for (model, name), color in zip(models, colors):
        DecisionBoundaryDisplay.from_estimator(
            model,
            X,
            plot_method="contour",
            colors=color,
            levels=[0],
            linestyles=["-"],
            ax=ax,
        )
        labels.append(mlines.Line2D([], [], color=color, label=name))

    plt.legend(handles=labels, loc="upper right")
    plt.title(title)

    filename = f"results/task_1_{title.replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.show()


def run_experiment(n_samples_1, n_samples_2, custom_weight):
    title = f"SVC {n_samples_1} vs {n_samples_2}"

    print(f"Experiment: class 0 = {n_samples_1}, class 1 = {n_samples_2}")

    X, y = generate_data(n_samples_1, n_samples_2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    svc = train_model(X_train, y_train)
    balanced_svc = train_model(X_train, y_train, class_weight="balanced")
    custom_svc = train_model(X_train, y_train, class_weight=custom_weight)

    evaluate_model(svc, X_test, y_test, "No weights")
    evaluate_model(balanced_svc, X_test, y_test, "Balanced")
    evaluate_model(custom_svc, X_test, y_test, f"Custom {custom_weight}")

    plot_results(
        X,
        y,
        [
            (svc, "SVC No Weights"),
            (balanced_svc, "SVC Balanced"),
            (custom_svc, "SVC Custom Weights"),
        ],
        title=title,
    )


def main():
    os.makedirs("results", exist_ok=True)

    experiments = [
        (1000, 10, {0: 1, 1: 100}),
        (1000, 100, {0: 1, 1: 10}),
        (1000, 500, {0: 1, 1: 2}),
        (1000, 1000, {0: 1, 1: 1}),
    ]

    for n1, n2, weight in experiments:
        run_experiment(n1, n2, weight)


if __name__ == "__main__":
    main()
