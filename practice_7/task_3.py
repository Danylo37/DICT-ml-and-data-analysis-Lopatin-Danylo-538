from sklearn.model_selection import cross_validate
from utils import plot_decision_regions
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import os


def main():
    os.makedirs("results", exist_ok=True)

    features, target = make_moons(n_samples=200, random_state=42, noise=None)

    plt.scatter(features[:, 0], features[:, 1], c=target)
    plt.title("Original Data")
    plt.tight_layout()
    plt.savefig("results/task_3_original.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    models = {'Linear': SVC(kernel="linear", random_state=42, C=1),
              'RBF': SVC(kernel="rbf", random_state=42, gamma=1, C=1),
              'Polynomial': SVC(kernel="poly", random_state=42, degree=3, gamma=1, C=1),
              'Sigmoid': SVC(kernel="sigmoid", random_state=42, gamma=1, C=1)}

    metrics = ['accuracy', 'precision', 'recall', 'f1']

    for name, model in models.items():
        print(name, end=':\n')

        model.fit(features, target)

        plot_decision_regions(features, target, classifier=model)
        plt.title(f"SVM Decision Regions - {name} Kernel")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"results/task_3_{name.lower()}.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        cv_results = cross_validate(model, features, target, scoring=metrics)

        for metric in metrics:
            scores = cv_results[f'test_{metric}']
            print(f"\t{metric}: {scores.mean()}")
        print()


if __name__ == '__main__':
    main()
