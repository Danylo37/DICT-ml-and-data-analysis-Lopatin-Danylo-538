from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
from sklearn import datasets
from sklearn import tree
import pydotplus
import os

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

iris = datasets.load_iris()
features = iris.data
target = iris.target

decisiontree = DecisionTreeClassifier(criterion="entropy", random_state=42)
model = decisiontree.fit(features, target)

observation = [[5, 4, 3, 2], [1, 1, 2, 3]]

print("Predictions:")
print(model.predict(observation))

print("Prediction probabilities:")
print(model.predict_proba(observation))

dot_data = tree.export_graphviz(
    decisiontree,
    out_file=None,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
)

graph = pydotplus.graph_from_dot_data(dot_data)

Image(graph.create_png())

graph.write_png(os.path.join(RESULTS_DIR, "task_1_decision_tree.png"))
