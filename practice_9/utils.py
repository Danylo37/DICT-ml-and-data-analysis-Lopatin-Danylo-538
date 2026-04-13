import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import imageio.v2 as imageio
from PIL import Image


def load_dataset():
    train_dataset = h5py.File("./input/train_catvnoncat.h5", "r")
    test_dataset = h5py.File("./input/test_catvnoncat.h5", "r")

    X_train = np.array(train_dataset["train_set_x"][:])
    y_train = np.array(train_dataset["train_set_y"][:]).reshape(1, -1)

    X_test = np.array(test_dataset["test_set_x"][:])
    y_test = np.array(test_dataset["test_set_y"][:]).reshape(1, -1)

    classes = np.array(test_dataset["list_classes"][:])

    return X_train, y_train, X_test, y_test, classes


def preprocess(X_train, X_test):
    X_train_flat = X_train.reshape(X_train.shape[0], -1).T / 255.0
    X_test_flat = X_test.reshape(X_test.shape[0], -1).T / 255.0
    return X_train_flat, X_test_flat


def show_and_save_plot(title, x, y, xlabel, ylabel, legend_label=None):
    plt.figure()
    if legend_label is None:
        plt.plot(x, y)
    else:
        plt.plot(x, y, label=legend_label)
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join("results", f"task_1_{title}.png"))
    plt.show()


def save_plot(title, x, ys, labels, xlabel, ylabel):
    plt.figure()
    for y, label in zip(ys, labels):
        plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"results/task_2_{title}.png")
    plt.show()


def plot_accuracy_comparison(custom_train_acc, custom_test_acc, sk_train_acc, sk_test_acc):
    labels = ["Custom Train", "Custom Test", "Sklearn Train", "Sklearn Test"]
    values = [custom_train_acc, custom_test_acc, sk_train_acc, sk_test_acc]

    plt.figure()
    plt.bar(labels, values)
    plt.ylabel("Accuracy (%)")
    plt.title("accuracy_comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "task_1_accuracy_comparison.png"))
    plt.show()


def predict_image(model, classes, img_path, num_px=64):
    image = np.array(imageio.imread(img_path))
    my_image = np.array(Image.fromarray(image).resize(size=(num_px, num_px))).reshape(
        (1, num_px * num_px * 3)).T / 255.0
    my_predicted_image = model.predict_proba(my_image)

    predicted_value = np.squeeze(my_predicted_image)
    predicted_class = classes[int(predicted_value)].decode("utf-8")

    plt.figure()
    plt.imshow(image)
    plt.title(f"Prediction: {predicted_class}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join("results", "task_1_prediction.png"))
    plt.show()

    print(f"y = {predicted_value}, your algorithm predicts a \"{predicted_class}\" picture.")

