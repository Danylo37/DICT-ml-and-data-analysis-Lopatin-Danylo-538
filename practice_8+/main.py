from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from scipy.stats import loguniform
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from time import time
import os


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(" ", 1)[-1]
    true_name = target_names[y_test[i]].rsplit(" ", 1)[-1]
    return "predicted: %s\ntrue:    %s" % (pred_name, true_name)


def main():
    os.makedirs("results", exist_ok=True)

    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    n_samples, h, w = lfw_people.images.shape

    X = lfw_people.data
    n_features = X.shape[1]

    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_components_list = [15, 50, 100, 150, 170, 200, 300]

    for n_components in n_components_list:
        print(
            "\nExtracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0])
        )
        t0 = time()
        pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True).fit(X_train)
        print("done in %0.3fs" % (time() - t0))

        eigenfaces = pca.components_.reshape((n_components, h, w))

        print("Projecting the input data on the eigenfaces orthonormal basis")
        t0 = time()
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        print("done in %0.3fs" % (time() - t0))

        print("\nFitting the classifier to the training set")
        t0 = time()
        param_grid = {
            "C": loguniform(1e3, 1e5),
            "gamma": loguniform(1e-4, 1e-1),
        }
        clf = RandomizedSearchCV(
            SVC(kernel="rbf", class_weight="balanced"), param_grid, n_iter=10
        )
        clf = clf.fit(X_train_pca, y_train)
        print("done in %0.3fs" % (time() - t0))
        print("Best estimator found by grid search:")
        print(clf.best_estimator_)

        print("\nPredicting people's names on the test set")
        t0 = time()
        y_pred = clf.predict(X_test_pca)
        print("done in %0.3fs" % (time() - t0))

        print(classification_report(y_test, y_pred, target_names=target_names))
        ConfusionMatrixDisplay.from_estimator(
            clf, X_test_pca, y_test, display_labels=target_names, xticks_rotation="vertical"
        )
        plt.tight_layout()
        plt.savefig(f"results/confusion_matrix_n_components_{n_components}.png", dpi=100, bbox_inches='tight')
        plt.show()

        prediction_titles = [
            title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])
        ]

        plot_gallery(X_test, prediction_titles, h, w)
        plt.savefig(f"results/predictions_n_components_{n_components}.png", dpi=100, bbox_inches='tight')
        plt.show()

        eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
        plot_gallery(eigenfaces, eigenface_titles, h, w)
        plt.savefig(f"results/eigenfaces_n_components_{n_components}.png", dpi=100, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    main()
