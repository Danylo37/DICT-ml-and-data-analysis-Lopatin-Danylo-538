import tensorflow as tf

from utils import (
    plot_batch,
    plot_class_balance,
    plot_average_images
)

NUM_CLASSES = 10
TASK_N = 1

FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def inspect_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    print("MNIST:")
    print(f"Train: {len(y_train)} samples")
    print(f"Test:  {len(y_test)} samples")
    print(f"Shape: {x_train.shape}")

    dataset_name = "mnist"
    plot_batch(x_train, y_train, TASK_N, dataset_name)
    plot_class_balance(y_train, NUM_CLASSES, TASK_N, dataset_name)
    plot_average_images(x_train, y_train, NUM_CLASSES, TASK_N, dataset_name)


def inspect_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    print("Fashion MNIST:")
    print(f"Train: {len(y_train)} samples")
    print(f"Test:  {len(y_test)} samples")
    print(f"Shape: {x_train.shape}")

    dataset_name = "fashion_mnist"
    plot_batch(x_train, y_train, TASK_N, dataset_name, class_names=FASHION_MNIST_CLASSES)
    plot_class_balance(y_train, NUM_CLASSES, TASK_N, dataset_name, class_names=FASHION_MNIST_CLASSES)
    plot_average_images(x_train, y_train, NUM_CLASSES, TASK_N, dataset_name, class_names=FASHION_MNIST_CLASSES)


def main():
    inspect_mnist()
    print()
    inspect_fashion_mnist()


if __name__ == "__main__":
    main()
