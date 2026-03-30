from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import os


def prepare_data(n_class_0, n_class_1):
    iris = load_iris()

    X = iris.data[:100, :2]
    y = iris.target[:100]

    X0 = X[y == 0][:n_class_0]
    X1 = X[y == 1][:n_class_1]

    X = np.vstack((X0, X1))
    y = np.array([0] * len(X0) + [1] * len(X1))

    return X, y


def train_model(X_train, y_train, class_weight=None):
    model = LogisticRegression(class_weight=class_weight, random_state=0)
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
    colors = ["k", "r"]
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

    filename = f"results/task_2_{title.replace(' ', '_')}.png"
    plt.savefig(filename)

    plt.show()
    plt.close()


def main():
    os.makedirs("results", exist_ok=True)

    n_class_0 = 50
    n_class_1 = 15

    X, y = prepare_data(n_class_0=n_class_0, n_class_1=n_class_1)

    title = f"LogReg Iris {n_class_0} vs {n_class_1}"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    lr = train_model(X_train, y_train)
    balanced_lr = train_model(X_train, y_train, class_weight="balanced")

    evaluate_model(lr, X_test, y_test, "No weights")
    evaluate_model(balanced_lr, X_test, y_test, "Balanced")

    plot_results(
        X,
        y,
        [
            (lr, "LogReg No Weights"),
            (balanced_lr, "LogReg Balanced"),
        ],
        title=title,
    )


if __name__ == "__main__":
    main()
