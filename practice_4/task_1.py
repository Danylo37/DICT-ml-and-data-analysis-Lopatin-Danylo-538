import os

import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def func(x):
    return np.cos(2 * np.pi * x)


def main():
    np.random.seed(42)

    n_samples = 20

    X = np.sort(np.random.rand(n_samples))

    noise_multipliers = [0.1, 0.5, 1.0, 2.0]
    degrees = [2, 3, 5, 9]

    results = []

    for col, degree in enumerate(degrees):
        fig, axes = plt.subplots(
            1, len(noise_multipliers),
            figsize=(5 * len(noise_multipliers), 4),
        )

        for row, noise in enumerate(noise_multipliers):
            y = func(X) + np.random.randn(n_samples) * noise

            ax = axes[row]
            ax.set_xticks([])
            ax.set_yticks([])

            polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
            linear_regression = LinearRegression()
            pipeline = Pipeline(
                [
                    ("polynomial", polynomial_features),
                    ("linear", linear_regression),
                ]
            )
            pipeline.fit(X[:, np.newaxis], y)

            scores = cross_val_score(
                pipeline, X[:, np.newaxis], y, scoring="neg_mean_squared_error", cv=10
            )
            mean_mse = -scores.mean()
            std_mse = scores.std()
            results.append((degree, noise, mean_mse, std_mse))

            X_test = np.linspace(0, 1, 100)
            ax.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
            ax.plot(X_test, func(X_test), label="True")
            ax.scatter(X, y, edgecolor="b", s=20, label="Data")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_xlim((0, 1))
            ax.set_ylim((-3, 3))
            ax.legend(loc="best")
            ax.set_title(
                f"noise×{noise}\nMSE = {mean_mse:.2e} (±{std_mse:.2e})",
            )

        fig.suptitle(f"Polynomial regression: degree {degree}")
        plt.tight_layout()

        plt.savefig(f"results/task_1_noise_vs_degree_{degree}.png")
        plt.show()

    results_sorted = sorted(results, key=lambda r: r[2])

    print("Results:")
    print(f"{'#':>3}  {'Degree':>6}  {'Noise':>5}  {'Mean MSE':>12}  {'Std MSE':>12}")
    for rank, (degree, noise, mean_mse, std_mse) in enumerate(results_sorted, 1):
        print(f"{rank:>3}  {degree:>6}  {noise:>5}  {mean_mse:>12.4f}  {std_mse:>12.4f}")


if __name__ == "__main__":
    main()
