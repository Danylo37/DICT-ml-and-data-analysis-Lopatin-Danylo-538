from sklearn.model_selection import train_test_split
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

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

max_depths = [2, 4, 6, 8]

for max_depth in max_depths:
    decisiontree = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, random_state=42)
    model = decisiontree.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test) * 100
    print(f"max_depth={max_depth}: {accuracy:.2f}%")

    dot_data = tree.export_graphviz(
        decisiontree,
        out_file=None,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
    )

    graph = pydotplus.graph_from_dot_data(dot_data)

    Image(graph.create_png())

    graph.write_png(os.path.join(RESULTS_DIR, f"task_3_decision_tree_depth={max_depth}.png"))
