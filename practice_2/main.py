from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


def visualize_sample_images(digits, output_dir, n_samples=10):
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()

    for i in range(n_samples):
        axes[i].imshow(digits.images[i], cmap='gray')
        axes[i].set_title(f'Label: {digits.target[i]}')
        axes[i].axis('off')

    plt.suptitle('Sample Images from Digits Dataset', fontsize=16, y=1.02)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/sample_images.png', bbox_inches='tight', dpi=100)
    plt.show()
    plt.close()


def class_distribution(digits, output_dir):
    classes, counts = np.unique(digits.target, return_counts=True)

    plt.figure(figsize=(6, 4))
    plt.bar(classes, counts)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution in digits Dataset')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, count in enumerate(counts):
        plt.text(i, count + 1, str(count), ha='center', fontsize=12)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/class_distribution.png', bbox_inches='tight', dpi=100)
    plt.show()
    plt.close()


def main():
    # Dataset
    digits = load_digits()

    print(f"Targets:\n{digits.target}\n")
    print(f"Features:\n{digits.feature_names[:10]}...\n")

    output_dir = os.path.join("practice_2", "results")
    os.makedirs(output_dir, exist_ok=True)

    visualize_sample_images(digits, output_dir)

    # Classification
    class_distribution(digits, output_dir)

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
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap="Blues")
        plt.title(name)

        filename = name.lower().replace(' ', '_')
        plt.savefig(f'{output_dir}/confusion_matrix_{filename}.png', bbox_inches='tight', dpi=100)
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()
