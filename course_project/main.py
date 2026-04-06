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


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # LOAD DATA

    df = pd.read_csv("data.csv")

    # ANALYZE DATE

    print("First 5 rows of data:")
    print(df.head())

    print("\nData info:")
    print(df.info())

    null_colls = df.isnull().sum()[df.isnull().sum() > 0]

    print("\nColumns with nulls:")
    print(null_colls.sort_values(ascending=False))
    print(f"Total: {null_colls.count()}")

    # REMOVE REDUNDANT COLUMNS

    df = df.drop(columns=["ID"])

    # ENCODE CATEGORICAL VARIABLES

    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    df = pd.get_dummies(df, columns=cat_cols.tolist())

    print("\nData info after preprocessing:")
    print(df.info())

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

    # TRAIN AND EVALUATE LINEAR REGRESSION

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred_lr = lr.predict(X_test)
    mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    print(
        f"\nLinear Regression:\n\tMAPE: {mape_lr * 100:.2f}%\n\tR²: {r2_lr * 100:.2f}%"
    )

    # TRAIN AND EVALUATE LASSO REGRESSION

    lasso = Lasso(alpha=1, random_state=42, max_iter=10000)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)

    mape_lasso = mean_absolute_percentage_error(y_test, y_pred_lasso)
    r2_lasso = r2_score(y_test, y_pred_lasso)

    print(f"\nLasso:\n\tMAPE: {mape_lasso * 100:.2f}%\n\tR²: {r2_lasso * 100:.2f}%")

    # TRAIN AND EVALUATE RIDGE REGRESSION

    ridge = Ridge(alpha=1, random_state=42)

    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)

    mape_ridge = mean_absolute_percentage_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)

    print(f"\nRidge:\n\tMAPE: {mape_ridge * 100:.2f}%\n\tR²: {r2_ridge * 100:.2f}%\n")

    # COMPARE MODELS

    comparison = pd.DataFrame(
        {
            "Model": ["Linear Regression", "Lasso", "Ridge"],
            "MAPE": [mape_lr, mape_lasso, mape_ridge],
            "R²": [r2_lr, r2_lasso, r2_ridge],
        }
    )
    print(comparison)


if __name__ == "__main__":
    main()
