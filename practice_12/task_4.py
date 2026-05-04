from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np
import os

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

features, target = make_classification(
    n_samples=10000,
    n_features=10,
    n_classes=2,
    weights=[0.9, 0.1],
    random_state=42
)

unique, counts = np.unique(target, return_counts=True)

plt.figure()
plt.bar(unique, counts)
plt.xticks(unique, [f"Class {i}" for i in unique])
plt.title("Class distribution")
plt.savefig(os.path.join(RESULTS_DIR, "task_4_class_distribution.png"))
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=42, stratify=target
)

random_forest = RandomForestClassifier(
    criterion="entropy", random_state=42, n_estimators=123
)
model = random_forest.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Random Forest Classifier without weights:")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

random_forest_weighted = RandomForestClassifier(
    criterion="entropy",
    random_state=42,
    n_estimators=123,
    class_weight="balanced",
)
model_weighted = random_forest_weighted.fit(X_train, y_train)

y_pred_weighted = model_weighted.predict(X_test)

print("\nRandom Forest Classifier with weights:")
print(classification_report(y_test, y_pred_weighted))
print(confusion_matrix(y_test, y_pred_weighted))
