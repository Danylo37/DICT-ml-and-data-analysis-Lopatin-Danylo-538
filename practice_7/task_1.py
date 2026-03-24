from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
import os


def main():
    os.makedirs("results", exist_ok=True)

    np.random.seed(42)
    features = np.random.randn(1000, 2)
    target_xor = np.logical_xor(features[:, 0] > 0, features[:, 1] > 0)
    target = np.where(target_xor, 0, 1)

    plt.subplots()
    plt.title("Original Data")
    plt.scatter(features[:, 0], features[:, 1], c=target)
    plt.savefig("results/task_1_original.png", dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
    plt.close()

    degrees = [2, 3, 4, 5]
    
    for degree in degrees:
        poly_svc = SVC(kernel="poly", random_state=42, degree=degree, gamma=1, C=1)
        poly_svc.fit(features, target)
        
        predictions = poly_svc.predict(features)
        
        accuracy = accuracy_score(target, predictions)
        precision = precision_score(target, predictions)
        recall = recall_score(target, predictions)
        f1 = f1_score(target, predictions)

        print(f"Degree: {degree}")
        print(f"\tAccuracy:  {accuracy:.4f}")
        print(f"\tPrecision: {precision:.4f}")
        print(f"\tRecall:    {recall:.4f}")
        print(f"\tF1-Score:  {f1:.4f}\n")

        plot_decision_regions(features, target, classifier=poly_svc)
        plt.title(f"Polynomial SVM (degree={degree})")
        plt.axis("off")
        plt.savefig(f"results/task_1_poly_degree_{degree}.png", dpi=300, bbox_inches='tight')
        plt.tight_layout()
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()