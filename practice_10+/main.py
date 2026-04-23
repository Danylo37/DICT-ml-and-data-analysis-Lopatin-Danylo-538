import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

RESULTS_DIR = "results"
BATCH_SIZE = 128
EPOCHS = 20
TARGET_ACCURACY = 0.95

CLASS_NAMES = [
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

ARCHITECTURES = [
    [8],
    [16],
    [16, 32],
    [16, 32, 32],
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


def rescale(data):
    return np.reshape(data, data.shape + (1,))


def format_label(label):
    return f"{label} ({CLASS_NAMES[label]})"


def model_name(hidden_units):
    if not hidden_units:
        return "no_hidden_layers"
    return "dense_" + "_".join(map(str, hidden_units))


def dense_classifier(input_shape, n_classes, hidden_units):
    layers = [tf.keras.layers.Input(shape=input_shape), tf.keras.layers.Flatten()]

    for units in hidden_units:
        layers.append(tf.keras.layers.Dense(units, activation=tf.nn.relu))

    layers.append(tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax))

    model = tf.keras.models.Sequential(layers)
    model.compile(
        optimizer="Adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_single_sample(image, label, save_path):
    fig = plt.figure()
    plt.title(f"Fashion MNIST sample - {format_label(label)}")
    plt.grid(False)
    plt.gray()
    plt.axis("off")
    plt.imshow(image)
    fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_batch(images, labels, save_path, n_rows=2, n_cols=5):
    fig = plt.gcf()
    fig.set_size_inches(n_cols * 4, n_rows * 4)
    fig.suptitle("Fashion MNIST samples", fontsize=16)

    for i in range(n_rows * n_cols):
        sp = plt.subplot(n_rows, n_cols, i + 1)
        sp.axis("off")
        plt.gray()
        plt.grid(False)
        plt.title(format_label(labels[i]))
        plt.imshow(images[i])

    fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_prediction(image, true_label, predicted_label, save_path):
    fig = plt.figure()
    plt.title(
        "Prediction - true: "
        f"{format_label(true_label)}, predicted: {format_label(predicted_label)}"
    )
    plt.grid(False)
    plt.gray()
    plt.axis("off")
    plt.imshow(image)
    fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_history(history, save_path, title):
    loss_values = history.history["loss"]
    accuracy_values = history.history["accuracy"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    fig.suptitle(title, fontsize=16)

    ax1.plot(range(len(loss_values)), loss_values)
    ax1.set_title("Loss")
    ax1.set_ylabel("Loss")
    ax1.grid(True)

    ax2.plot(range(len(accuracy_values)), accuracy_values)
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def train_and_evaluate_model(
    input_shape,
    n_classes,
    hidden_units,
    training_images,
    training_labels,
    test_images,
    test_labels,
):
    name = model_name(hidden_units)
    model_dir = os.path.join(RESULTS_DIR, name)
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Model: {name}")
    print(f"Hidden layers: {hidden_units}")
    print(f"{'=' * 60}")

    model = dense_classifier(input_shape, n_classes, hidden_units)

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
        "name": name,
        "model": model,
        "history": history,
        "classifications": classifications,
        "loss": loss,
        "accuracy": accuracy,
        "model_dir": model_dir,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    data = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = data.load_data()

    index = 4556
    print("Data:")
    print(f"Label: {format_label(training_labels[index])}")
    print(f"\nImage pixel array:\n {training_images[index]}")

    plot_single_sample(
        training_images[index],
        training_labels[index],
        os.path.join(RESULTS_DIR, "sample.png"),
    )

    plot_batch(
        training_images,
        training_labels,
        os.path.join(RESULTS_DIR, "batch.png"),
    )

    training_images = training_images / 255.0
    test_images = test_images / 255.0

    print("\nShape of training images:")
    print(training_images.shape)
    print(test_images.shape)

    training_images = rescale(training_images)
    test_images = rescale(test_images)

    print("\nShape of training images after rescaling:")
    print(training_images.shape)
    print(test_images.shape)

    input_shape = training_images[0].shape
    n_classes = 10

    results = []

    for hidden_units in ARCHITECTURES:
        result = train_and_evaluate_model(
            input_shape=input_shape,
            n_classes=n_classes,
            hidden_units=hidden_units,
            training_images=training_images,
            training_labels=training_labels,
            test_images=test_images,
            test_labels=test_labels,
        )

        results.append(result)

        model_dir = result["model_dir"]
        plot_history(
            result["history"],
            os.path.join(model_dir, "history.png"),
            title=f"Training history - {result['name']}",
        )

    best_result = max(results, key=lambda x: x["accuracy"])
    print("\nBest model:")
    print(f"Name: {best_result['name']}")
    print(f"Test accuracy: {best_result['accuracy']:.4f}")
    print(f"Test loss: {best_result['loss']:.4f}")

    best_classifications = best_result["classifications"]

    index = 1
    print("\nTesting images:")
    print(f"True label: {format_label(test_labels[index])}")
    print(f"Predicted label: {format_label(np.argmax(best_classifications[index]))}")

    wrong_index = None
    for i in range(test_images.shape[0]):
        if test_labels[i] != np.argmax(best_classifications[i]):
            wrong_index = i
            print(f"\nWrong prediction (index={wrong_index}):")
            print(f"True label: {format_label(test_labels[wrong_index])}")
            print(
                f"Predicted label: "
                f"{format_label(np.argmax(best_classifications[wrong_index]))}"
            )
            break

    if wrong_index is not None:
        plot_prediction(
            test_images[wrong_index],
            test_labels[wrong_index],
            np.argmax(best_classifications[wrong_index]),
            os.path.join(best_result["model_dir"], "prediction.png"),
        )


if __name__ == "__main__":
    main()

