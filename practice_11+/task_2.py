import numpy as np
import tensorflow as tf

from utils import (
    plot_batch,
    plot_class_balance,
    plot_average_images,
    plot_prediction,
    plot_history,
)

TASK_N = 2
DATASET_NAME = "cifar10"
NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 20
TARGET_ACCURACY = 0.95

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self, target_accuracy=TARGET_ACCURACY):
        super().__init__()
        self.target_accuracy = target_accuracy

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if logs.get("accuracy", 0) >= self.target_accuracy:
            print("\nWe've got the desired accuracy")
            self.model.stop_training = True


def format_label(label):
    return f"{label} ({CLASS_NAMES[label]})"


def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate_model(
    input_shape,
    training_images,
    training_labels,
    test_images,
    test_labels,
):
    model = build_model(input_shape)

    print("\nModel summary:")
    model.summary()

    callback = MyCallback()

    print("\nModel fitting:")
    history = model.fit(
        training_images,
        training_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[callback],
        verbose=1,
    )

    print("\nModel evaluation:")
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

    classifications = model.predict(test_images, verbose=0)

    return {
        "model": model,
        "history": history,
        "classifications": classifications,
        "loss": loss,
        "accuracy": accuracy,
    }


def main():
    (training_images, training_labels), (test_images, test_labels) = (
        tf.keras.datasets.cifar10.load_data()
    )

    training_labels = training_labels.flatten()
    test_labels = test_labels.flatten()

    print("CIFAR-10:")
    print(f"Train: {len(training_labels)} samples")
    print(f"Test:  {len(test_labels)} samples")
    print(f"Shape: {training_images.shape}")

    plot_batch(training_images, training_labels, TASK_N, DATASET_NAME, class_names=CLASS_NAMES)
    plot_class_balance(training_labels, NUM_CLASSES, TASK_N, DATASET_NAME, class_names=CLASS_NAMES)
    plot_average_images(training_images, training_labels, NUM_CLASSES, TASK_N, DATASET_NAME, class_names=CLASS_NAMES)

    training_images = training_images / 255.0
    test_images = test_images / 255.0

    input_shape = training_images[0].shape

    result = train_and_evaluate_model(
        input_shape=input_shape,
        training_images=training_images,
        training_labels=training_labels,
        test_images=test_images,
        test_labels=test_labels,
    )

    plot_history(
        result["history"],
        TASK_N,
        DATASET_NAME,
        title="history",
    )

    classifications = result["classifications"]

    index = 1
    print("\nSample prediction:")
    print(f"True label: {format_label(test_labels[index])}")
    print(f"Predicted label: {format_label(np.argmax(classifications[index]))}")

    for i in range(test_images.shape[0]):
        if test_labels[i] != np.argmax(classifications[i]):
            print(f"\nFirst wrong prediction (index={i}):")
            print(f"True label: {format_label(test_labels[i])}")
            print(f"Predicted label: {format_label(np.argmax(classifications[i]))}")

            plot_prediction(
                test_images[i],
                test_labels[i],
                np.argmax(classifications[i]),
                TASK_N,
                DATASET_NAME,
                class_names=CLASS_NAMES,
            )
            break


if __name__ == "__main__":
    main()
