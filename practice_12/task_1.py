from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from IPython.display import Image
from sklearn import datasets
from sklearn import tree
import numpy as np
import pydotplus
import os

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

housing = datasets.fetch_california_housing()
features = housing.data
target = housing.target

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

decisiontree = DecisionTreeRegressor(random_state=42, max_depth=5, min_samples_split=10)
model = decisiontree.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, predictions) * 100

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")

dot_data = tree.export_graphviz(
    decisiontree,
    out_file=None,
    feature_names=housing.feature_names,
    filled=True,
    rounded=True,
    max_depth=3
)

graph = pydotplus.graph_from_dot_data(dot_data)

Image(graph.create_png())

graph.write_png(os.path.join(RESULTS_DIR, "task_1_decision_tree.png"))
