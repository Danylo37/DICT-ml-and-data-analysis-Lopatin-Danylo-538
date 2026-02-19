from sklearn.datasets import make_classification
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate, train_test_split, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def create_visualizations(dataset, results_dir):
    sns.set_style('whitegrid')
    metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    models = dataset['model'].unique()

    params = [('n_features', 'Number of Features'), ('n_informative', 'Number of Informative Features'),
              ('n_redundant', 'Number of Redundant Features'), ('n_classes', 'Number of Classes')]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Accuracy vs Dataset Parameters', fontsize=16, fontweight='bold')

    for idx, (param, name) in enumerate(params):
        ax = axes[idx // 2, idx % 2]
        for model in models:
            data = dataset[dataset['model'] == model]
            ax.scatter(data[param], data['accuracy'], label=model, alpha=0.7, s=100)
        ax.set_title(f'Accuracy vs {name}', fontweight='bold')
        ax.set_xlabel(name)
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / 'accuracy_vs_parameters.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('All Metrics by Model', fontsize=16, fontweight='bold')

    for idx, model in enumerate(models):
        data = dataset[dataset['model'] == model]
        means = [data[m].mean() for m in metrics]
        bars = axes[idx].bar(metric_names, means, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[idx].set_title(model, fontweight='bold')
        axes[idx].set_ylabel('Score')
        axes[idx].set_ylim(0, 1)
        axes[idx].grid(axis='y', alpha=0.3)

        for bar in bars:
            h = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2, h, f'{h:.3f}',
                          ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(results_dir / 'all_metrics_by_model.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Accuracy Heatmap: Classes vs Features', fontsize=16, fontweight='bold')

    for idx, model in enumerate(models):
        pivot = dataset[dataset['model'] == model].pivot_table(
            values='accuracy', index='n_classes', columns='n_features', aggfunc='mean')
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[idx],
                   cbar_kws={'label': 'Accuracy'}, vmin=0, vmax=1)
        axes[idx].set_title(model, fontweight='bold')
        axes[idx].set_xlabel('Number of Features')
        axes[idx].set_ylabel('Number of Classes')

    plt.tight_layout()
    plt.savefig(results_dir / 'accuracy_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def task_1_dataset_parameters():
    print("TASK 1: Dataset Parameters vs Model Metrics")

    dataset_configs = [
        {'n_features': 5, 'n_informative': 3, 'n_redundant': 0, 'n_classes': 2},
        {'n_features': 10, 'n_informative': 5, 'n_redundant': 2, 'n_classes': 2},
        {'n_features': 10, 'n_informative': 7, 'n_redundant': 0, 'n_classes': 3},
        {'n_features': 15, 'n_informative': 8, 'n_redundant': 3, 'n_classes': 4},
        {'n_features': 20, 'n_informative': 10, 'n_redundant': 5, 'n_classes': 2},
        {'n_features': 8, 'n_informative': 5, 'n_redundant': 1, 'n_classes': 3},
        {'n_features': 12, 'n_informative': 8, 'n_redundant': 2, 'n_classes': 2},
        {'n_features': 25, 'n_informative': 15, 'n_redundant': 5, 'n_classes': 5},
        {'n_features': 18, 'n_informative': 12, 'n_redundant': 3, 'n_classes': 3},
        {'n_features': 30, 'n_informative': 20, 'n_redundant': 10, 'n_classes': 6},
    ]

    models = {
        'Dummy Classifier': DummyClassifier(strategy='uniform', random_state=42),
        'Logistic Regression': make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=42)
        ),
        'Random Forest Classifier': RandomForestClassifier(random_state=42),
    }

    scorers = {
        'accuracy': 'accuracy',
        'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=0),
        'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=0),
        'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0)
    }

    results = []

    for i, config in enumerate(dataset_configs, 1):
        print(f"\nDataset {i}/{len(dataset_configs)}: {config}:")

        X, y = make_classification(
            n_samples=10000,
            n_features=config['n_features'],
            n_informative=config['n_informative'],
            n_redundant=config['n_redundant'],
            n_classes=config['n_classes'],
            random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        for name, model in models.items():
            print(f"\n{name}:")

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_results = cross_validate(model, X_train, y_train, scoring=scorers, cv=kf)

            result_row = {
                'n_features': config['n_features'],
                'n_informative': config['n_informative'],
                'n_redundant': config['n_redundant'],
                'n_classes': config['n_classes'],
                'model': name,
            }

            for metric_name in scorers.keys():
                scores = cv_results[f'test_{metric_name}']
                mean_score = scores.mean()
                result_row[metric_name] = mean_score
                print(f"\t{metric_name}: {mean_score:.4f}")

            results.append(result_row)

    df = pd.DataFrame(results)
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    df.to_csv(results_dir / 'model_comparison.csv', index=False)

    create_visualizations(df, results_dir)


def task_2_kfold_splits():
    print("TASK 2: Impact of KFold n_splits on Model Performance")

    X, y = make_classification(
        n_samples=10000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    n_splits_values = [2, 3, 5, 7, 10, 15, 20]

    models = {
        'Dummy Classifier': DummyClassifier(strategy='uniform', random_state=42),
        'Logistic Regression': make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=42)
        ),
        'Random Forest Classifier': RandomForestClassifier(random_state=42),
    }

    results = []

    for n_splits in n_splits_values:
        print(f"\nTesting with n_splits = {n_splits}")

        for name, model in models.items():
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_results = cross_validate(model, X_train, y_train, scoring='accuracy', cv=kf)

            mean_accuracy = cv_results['test_score'].mean()
            std_accuracy = cv_results['test_score'].std()

            results.append({
                'n_splits': n_splits,
                'model': name,
                'accuracy_mean': mean_accuracy,
                'accuracy_std': std_accuracy
            })

            print(f"  {name}: {mean_accuracy:.4f} (±{std_accuracy:.4f})")

    df = pd.DataFrame(results)
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    df.to_csv(results_dir / 'kfold_splits_comparison.csv', index=False)

    plt.figure(figsize=(12, 6))

    for model_name in df['model'].unique():
        model_data = df[df['model'] == model_name]
        plt.plot(model_data['n_splits'], model_data['accuracy_mean'],
                marker='o', label=model_name, linewidth=2, markersize=8)
        plt.fill_between(model_data['n_splits'],
                        model_data['accuracy_mean'] - model_data['accuracy_std'],
                        model_data['accuracy_mean'] + model_data['accuracy_std'],
                        alpha=0.2)

    plt.xlabel('Number of Splits (n_splits)', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Impact of KFold n_splits on Model Performance', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'kfold_splits_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def main():
    # Task 1: Dataset parameters vs metrics
    task_1_dataset_parameters()

    # Task 2: KFold n_splits comparison
    task_2_kfold_splits()


if __name__ == '__main__':
    main()
