import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge


def evaluate_regression(n_samples, n_features, noise=50, random_state=1):
    features, target = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(n_features, 3),
        n_targets=1,
        noise=noise,
        coef=False,
        random_state=random_state
    )

    ridge = Ridge(alpha=1.0)

    mse_score = cross_val_score(ridge, features, target, scoring='neg_mean_squared_error', cv=5)
    r2_score = cross_val_score(ridge, features, target, scoring='r2', cv=5)

    return np.mean(-mse_score), np.mean(r2_score)


def plot_metric(x_values, y_values, xlabel, ylabel, title, filename, color='red', marker='o'):
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker=marker, linewidth=2, markersize=8, color=color)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'results/task_1_{filename}', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def compare_n_samples(n_samples_range, n_features=3):
    results = {'mse': [], 'r2': []}

    print(f"Comparison of different n_samples (with n_features={n_features}):")

    for n_samp in n_samples_range:
        mean_mse, mean_r2 = evaluate_regression(n_samples=n_samp, n_features=n_features)

        results['mse'].append(mean_mse)
        results['r2'].append(mean_r2)

        print(f"n_samples={n_samp}: MSE={mean_mse:.2f}, R2={mean_r2:.4f}")

    return results


def compare_n_features(n_features_range, n_samples=100):
    results = {'mse': [], 'r2': []}

    print(f"\nComparison of different n_features (with n_samples={n_samples}):")

    for n_feat in n_features_range:
        mean_mse, mean_r2 = evaluate_regression(n_samples=n_samples, n_features=n_feat)

        results['mse'].append(mean_mse)
        results['r2'].append(mean_r2)

        print(f"n_features={n_feat}: MSE={mean_mse:.2f}, R2={mean_r2:.4f}")

    return results


def plot_samples_comparison(n_samples_range, results):
    plot_metric(
        n_samples_range, results['mse'],
        xlabel='Number of Samples',
        ylabel='Negative Mean Squared Error',
        title='MSE vs Number of Samples (n_features=3)',
        filename='mse_vs_n_samples.png',
    )

    plot_metric(
        n_samples_range, results['r2'],
        xlabel='Number of Samples',
        ylabel='R2 Score',
        title='R2 Score vs Number of Samples (n_features=3)',
        filename='r2_vs_n_samples.png',
    )


def plot_features_comparison(n_features_range, results):
    plot_metric(
        n_features_range, results['mse'],
        xlabel='Number of Features',
        ylabel='Mean Squared Error',
        title='MSE vs Number of Features (n_samples=100)',
        filename='mse_vs_n_features.png',
    )

    plot_metric(
        n_features_range, results['r2'],
        xlabel='Number of Features',
        ylabel='R2 Score',
        title='R2 Score vs Number of Features (n_samples=100)',
        filename='r2_vs_n_features.png',
    )


def main():
    n_samples_range = [50, 100, 200, 500, 1000]
    n_features_range = [2, 3, 5, 10, 20]

    os.makedirs("results", exist_ok=True)

    # Task 1: Compare different n_samples
    results_samples = compare_n_samples(n_samples_range, n_features=3)
    plot_samples_comparison(n_samples_range, results_samples)

    # Task 2: Compare different n_features
    results_features = compare_n_features(n_features_range, n_samples=1000)
    plot_features_comparison(n_features_range, results_features)


if __name__ == '__main__':
    main()
