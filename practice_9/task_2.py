from utils import load_dataset, preprocess, save_plot
from task_1 import CustomLogRegression
import numpy as np
import os


def experiment_learning_rates(X_train, y_train, X_test, y_test):
    learning_rates = [0.01, 0.005, 0.001, 0.0005]
    curves = []
    labels = []

    for lr in learning_rates:
        model = CustomLogRegression(learning_rate=lr, num_iterations=2000)
        model.fit(X_train, y_train)
        curves.append(model.costs)
        labels.append(f"lr={lr}")

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        print(f"lr={lr} train={train_acc:.2f}% test={test_acc:.2f}%")

    x = np.arange(len(curves[0])) * 100
    save_plot("learning_rate_comparison", x, curves, labels, "iterations", "cost")


def experiment_iterations(X_train, y_train, X_test, y_test):
    iterations_list = [200, 500, 1000, 2000, 4000]
    train_scores = []
    test_scores = []

    for it in iterations_list:
        model = CustomLogRegression(learning_rate=0.005, num_iterations=it)
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        train_scores.append(train_acc)
        test_scores.append(test_acc)

        print(f"iter={it} train={train_acc:.2f}% test={test_acc:.2f}%")

    save_plot(
        "iterations_vs_accuracy",
        iterations_list,
        [train_scores, test_scores],
        ["train", "test"],
        "iterations",
        "accuracy",
    )


def main():
    os.makedirs("results", exist_ok=True)

    X_train, y_train, X_test, y_test, _ = load_dataset()
    X_train, X_test = preprocess(X_train, X_test)

    experiment_learning_rates(X_train, y_train, X_test, y_test)
    print()
    experiment_iterations(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()