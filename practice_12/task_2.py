from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import os

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

iris = datasets.load_iris()
features = iris.data
target = iris.target

random_forest = RandomForestClassifier(
    criterion="entropy", random_state=42, n_estimators=123
)
model = random_forest.fit(features, target)

observation = [[5, 4, 3, 2], [1, 1, 2, 3]]
print("Predictions (all features):")
print(model.predict(observation))

print("Prediction probabilities (all features):")
print(model.predict_proba(observation))

importances = model.feature_importances_

indices = np.argsort(importances)[::-1]

names = [iris.feature_names[i] for i in indices]

plt.title("Features importance")
plt.bar(range(features.shape[1]), importances[indices])
plt.xticks(range(features.shape[1]), names, rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "task_2_feature_importance.png"))
plt.show()

selector = SelectFromModel(random_forest, threshold=0.3)

features_important = selector.fit_transform(features, target)

model = random_forest.fit(features_important, target)

observation = [[5, 4], [1, 1]]
print()
print("Predictions (selected features only):")
print(model.predict(observation))

print("Prediction probabilities (selected features only):")
print(model.predict_proba(observation))
