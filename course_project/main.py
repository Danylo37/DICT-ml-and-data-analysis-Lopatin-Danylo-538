from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

RESULTS_DIR = "results"
TARGET = "price"


def plot_predicted_vs_actual(y_test, y_pred, name):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, label="Predictions")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', label="Perfect Prediction")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{name}: Predicted vs Actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{name}_predicted_vs_actual.png"))
    plt.show()


def plot_model_comparison(comparison):
    plt.figure(figsize=(8, 5))
    sns.barplot(data=comparison, x="Model", y="R²")
    plt.title("Model Comparison (R²)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_comparison_r2.png"))
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=comparison, x="Model", y="MAPE")
    plt.title("Model Comparison (MAPE)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_comparison_mape.png"))
    plt.show()


def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{model_name}:\n\tMAPE: {mape * 100:.2f}%\n\tR²: {r2 * 100:.2f}%")

    plot_predicted_vs_actual(y_test, y_pred, model_name)

    return mape, r2, y_pred


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # LOAD DATA

    df: pd.DataFrame = pd.read_csv("data.csv")  # type: ignore

    # ANALYZE DATE

    print("First 5 rows of data:")
    print(df.head())

    print("\nData info:")
    df.info()

    null_cols = df.isnull().sum()[df.isnull().sum() > 0]

    print("\nColumns with nulls:")
    print(null_cols.sort_values(ascending=False))
    print(f"Total: {null_cols.count()}")

    # REMOVE REDUNDANT COLUMNS

    df = df.drop(columns=["ID"])

    # ENCODE CATEGORICAL VARIABLES

    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    df = pd.get_dummies(df, columns=cat_cols.tolist())

    print("\nData info after preprocessing:")
    df.info()

    # SPLIT DATA

    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # SCALE DATA

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # PLOT CORRELATION MATRIX

    corr = df.corr(numeric_only=True)

    target_corr = corr[TARGET].sort_values(ascending=False)

    top_features = target_corr.index[:15]

    plt.figure(figsize=(10, 8))
    sns.heatmap(df[top_features].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "correlation_matrix.png"))
    plt.show()

    # TRAIN AND EVALUATE MODELS

    lr = LinearRegression()
    mape_lr, r2_lr, y_pred_lr = train_and_evaluate_model(
        lr, X_train, X_test, y_train, y_test, "Linear"
    )

    lasso = Lasso(alpha=1, random_state=42, max_iter=10000)
    mape_lasso, r2_lasso, y_pred_lasso = train_and_evaluate_model(
        lasso, X_train, X_test, y_train, y_test, "Lasso"
    )

    ridge = Ridge(alpha=1, random_state=42)
    mape_ridge, r2_ridge, y_pred_ridge = train_and_evaluate_model(
        ridge, X_train, X_test, y_train, y_test, "Ridge"
    )

    # COMPARE MODELS

    comparison = pd.DataFrame(
        {
            "Model": ["Linear Regression", "Lasso", "Ridge"],
            "MAPE": [mape_lr, mape_lasso, mape_ridge],
            "R²": [r2_lr, r2_lasso, r2_ridge],
        }
    )
    print(comparison)

    # PLOT MODEL COMPARISON

    plot_model_comparison(comparison)


if __name__ == "__main__":
    main()
