from utils import plot_decision_boundary, load_planar_dataset
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt
from ann import ANNModel
import numpy as np
import os

np.random.seed(42)
os.makedirs("results", exist_ok=True)

X, Y = load_planar_dataset()

plt.figure(figsize=(8, 6))
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.get_cmap("Spectral"))
plt.title("Original")
plt.savefig("results/task_1_Original.png")
plt.show()

shape_X = X.shape
shape_Y = Y.shape
m = shape_X[1]

print("Dataset:")
print(f"The shape of X is: {str(shape_X)}")
print(f"The shape of Y is: {str(shape_Y)}")
print(f"Training examples: {m}")

clf = LogisticRegressionCV()
clf.fit(X.T, Y.ravel())

plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression Prediction")
plt.savefig("results/task_1_Logistic_Regression_pred.png")
plt.show()

LR_predictions = clf.predict(X.T)
Y_flat = Y.ravel()
acc = float((np.dot(Y_flat, LR_predictions) + np.dot(1 - Y_flat, 1 - LR_predictions))
            / float(Y_flat.size) * 100)

print(f"\nLogistic Regression Accuracy: {acc:.2f}% (percentage of correctly labelled datapoints)")

print("\nArtificial Neural Network:")

model = ANNModel(n_x=X.shape[0], n_h=4, n_y=1, learning_rate=1.2)
model.fit(X, Y, num_iterations=10000, print_cost=True)

print(f"\nANN Accuracy: {model.score(X, Y):.2f}%")

plot_decision_boundary(model.get_decision_boundary_function(), X, Y)
plt.title("ANN Decision Boundary Prediction (hidden layer size = 4)")
plt.savefig("results/task_1_ANN_pred.png")
plt.show()

print("\nTesting different hidden layer sizes:")
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50, 100]

fig, axes = plt.subplots(4, 2, figsize=(12, 14))
axes = axes.flatten()

for i, n_h in enumerate(hidden_layer_sizes):
    ax = axes[i]
    plt.sca(ax)

    model = ANNModel(n_x=X.shape[0], n_h=n_h, n_y=1, learning_rate=1.2)
    model.fit(X, Y, num_iterations=5000, print_cost=False)

    plot_decision_boundary(model.get_decision_boundary_function(), X, Y)
    accuracy = model.score(X, Y)

    ax.set_title(f'Hidden Layer Size = {n_h} ({accuracy:.2f}%)')
    print(f"Accuracy for {n_h} hidden units: {accuracy:.2f}%")

for j in range(len(hidden_layer_sizes), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()

plt.subplots_adjust(top=0.93)

plt.suptitle("Decision Boundaries for Different Hidden Layer Sizes")
plt.savefig("results/task_1_hidden_layers_comparison.png")
plt.show()
