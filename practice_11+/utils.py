import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results"


def _save_path(task_n, title):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, f"task_{task_n}_{title}.png")


def plot_batch(images, labels, task_n, dataset, class_names=None, n_rows=3, n_cols=5):
    fig = plt.gcf()
    fig.set_size_inches(n_cols * 4, n_rows * 4)
    fig.suptitle(f"Dataset samples ({dataset} dataset)", fontsize=16)

    for i in range(n_rows * n_cols):
        idx = np.random.randint(0, len(labels))
        sp = plt.subplot(n_rows, n_cols, i + 1)
        sp.axis("off")
        plt.grid(False)
        label = labels[idx]
        title = class_names[label] if class_names else str(label)
        plt.title(title)
        plt.imshow(images[idx])

    plt.tight_layout()
    fig.savefig(_save_path(task_n, f"{dataset}_batch"), bbox_inches="tight")
    plt.show()


def plot_class_balance(labels, num_classes, task_n, dataset, class_names=None):
    centers = np.arange(0, num_classes + 1)
    counts, _ = np.histogram(labels, bins=centers - 0.5)
    tick_labels = class_names if class_names else [str(i) for i in range(num_classes)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f"Class Balance ({dataset} dataset)", fontsize=16)

    ax1.bar(centers[:-1], counts)
    ax1.set_xticks(centers[:-1])
    ax1.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Count")
    ax1.set_title("Absolute counts")
    ax1.grid(True, axis="y")

    ax2.bar(centers[:-1], counts / np.sum(counts))
    ax2.set_xticks(centers[:-1])
    ax2.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax2.axhline(y=1 / num_classes, color="red", linestyle="--", label="Perfect balance")
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Fraction")
    ax2.set_title("Relative distribution")
    ax2.legend()
    ax2.grid(True, axis="y")

    plt.tight_layout()
    fig.savefig(_save_path(task_n, f"{dataset}_class_balance"), bbox_inches="tight")
    plt.show()


def plot_average_images(images, labels, num_classes, task_n, dataset, class_names=None):
    is_color = images.ndim == 4 and images.shape[-1] == 3
    n_cols = 5
    n_rows = int(np.ceil(num_classes / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    fig.suptitle(f"Average image per class ({dataset} dataset)", fontsize=16)
    axes = axes.flatten()

    for class_id in range(num_classes):
        mask = labels == class_id
        class_images = images[mask] / 255.0
        avg = np.mean(class_images, axis=0)

        if is_color:
            axes[class_id].imshow(np.clip(avg, 0, 1))
        else:
            axes[class_id].imshow(avg / np.sum(avg), cmap="gray")

        axes[class_id].axis("off")
        title = class_names[class_id] if class_names else str(class_id)
        axes[class_id].set_title(title)

    for idx in range(num_classes, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    fig.savefig(_save_path(task_n, f"{dataset}_average_images"), bbox_inches="tight")
    plt.show()


def plot_prediction(image, true_label, predicted_label, task_n, dataset, class_names=None):
    fig = plt.figure()
    true_title = class_names[true_label] if class_names else str(true_label)
    pred_title = class_names[predicted_label] if class_names else str(predicted_label)
    plt.title(f"True: {true_label} ({true_title}), Predicted: {predicted_label} ({pred_title})")
    plt.grid(False)
    plt.axis("off")
    plt.imshow(image)
    fig.savefig(_save_path(task_n, f"{dataset}_prediction"), bbox_inches="tight")
    plt.show()


def plot_history(history, task_n, dataset, title):
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
    fig.savefig(_save_path(task_n, f"{dataset}_{title.replace(' ', '_')}"), bbox_inches="tight")
    plt.show()
