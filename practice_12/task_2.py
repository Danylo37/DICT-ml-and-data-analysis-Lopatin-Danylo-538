from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import os


RESULTS_DIR = "results"


def create_model():
    return RandomForestRegressor(
        random_state=42,
        n_estimators=123,
        max_depth=10,
        min_samples_split=10,
    )


def evaluate_model(model, X_test, y_test, label=""):
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, predictions) * 100

    print(f"\n{label}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")


def plot_feature_importance(model, feature_names, features):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [feature_names[i] for i in indices]

    plt.title("Features importance")
    plt.bar(range(features.shape[1]), importances[indices])
    plt.xticks(range(features.shape[1]), names, rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "task_2_feature_importance.png"))
    plt.show()


def select_features(X_train, y_train, X_test, feature_names):
    selector = SelectFromModel(
        create_model(),
        threshold="mean"
    )

    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    selected_mask = selector.get_support()
    selected_feature_names = np.array(feature_names)[selected_mask]

    print("\nSelected features:")
    print(selected_feature_names)

    return X_train_selected, X_test_selected


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    data = datasets.fetch_california_housing()
    features, target, feature_names = data.data, data.target, data.feature_names
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = create_model()
    model.fit(X_train, y_train)

    print("Predictions (all features):")
    print(model.predict(X_test[:2]))

    evaluate_model(model, X_test, y_test, label="All features")

    plot_feature_importance(model, feature_names, features)

    X_train_sel, X_test_sel = select_features(X_train, y_train, X_test, feature_names)

    model_selected = create_model()
    model_selected.fit(X_train_sel, y_train)

    print("\nPredictions (selected features only):")
    print(model_selected.predict(X_test_sel[:2]))

    evaluate_model(model_selected, X_test_sel, y_test, label="Selected features")


if __name__ == "__main__":
    main()