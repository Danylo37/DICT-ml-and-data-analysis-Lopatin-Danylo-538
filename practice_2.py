from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


def class_distribution(digits):
    classes, counts = np.unique(digits.target, return_counts=True)

    plt.figure(figsize=(6, 4))
    plt.bar(classes, counts)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution in digits Dataset')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, count in enumerate(counts):
        plt.text(i, count + 1, str(count), ha='center', fontsize=12)

    plt.show()


def main():
    # Dataset
    digits = load_digits()

    print(f"Targets:\n{digits.target}\n")
    print(f"Features:\n{digits.feature_names[:10]}...\n")

    # Classification
    class_distribution(digits)

    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        'Dummy Classifier': DummyClassifier(),
        'Logistic Regression': make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000)
        ),
        'Random Forest Classifier': RandomForestClassifier(),
    }

    scorers = {
        'accuracy': 'accuracy',
        'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=0),
        'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=0),
        'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0)
    }

    for name, model in models.items():
        print(name)

        cv_results = cross_validate(model, X_train, y_train, scoring=scorers, cv=5)

        for metric_name in scorers.keys():
            scores = cv_results[f'test_{metric_name}']
            print(f"\t{metric_name}: {scores.mean()}")
        print()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm).plot(cmap="Blues")
        plt.title(name)
        plt.show()


if __name__ == '__main__':
    main()
