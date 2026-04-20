import numpy as np
import os

from utils import load_extra_datasets, save_original_plot, save_prediction_plot
from ann import ANNModel


def main():
    np.random.seed(42)
    os.makedirs("results", exist_ok=True)

    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

    datasets = {
        "noisy_circles": noisy_circles,
        "noisy_moons": noisy_moons,
        "blobs": blobs,
        "gaussian_quantiles": gaussian_quantiles,
    }

    for dataset_name, dataset in datasets.items():
        X, Y = dataset

        X = X.T
        Y = Y.reshape(1, -1)

        save_original_plot(X, Y, dataset_name)

        if dataset_name == "blobs":
            print("Skip blobs (not suitable for binary classification)")
            continue

        model = ANNModel(
            n_x=X.shape[0],
            n_h=4,
            n_y=1,
            learning_rate=1.2
        )
        model.fit(X, Y, num_iterations=10000, print_cost=False)

        accuracy = model.score(X, Y)
        print(f"Accuracy on {dataset_name}: {accuracy:.2f}%")

        save_prediction_plot(model, X, Y, dataset_name, accuracy)

if __name__ == '__main__':
    main()