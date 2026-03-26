from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

RESULTS_DIR = "results"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    #############
    # LOAD DATA #
    #############

    df = pd.read_csv('data.csv')

    ################
    # ANALYZE DATE #
    ################

    print("First 5 rows of data:")
    print(df.head())

    print("\nData info:")
    print(df.info())

    null_colls = df.isnull().sum()[df.isnull().sum() > 0]

    print("\nColumns with nulls:")
    print(null_colls.sort_values(ascending=False))
    print(f"Total: {null_colls.count()}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title("Heatmap of Missing Values")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "missing_values.png"))
    plt.show()

    ############################
    # REMOVE REDUNDANT COLUMNS #
    ############################

    df = df.drop(columns=["Order", "PID"])

    #######################
    # FILL MISSING VALUES #
    #######################

    # Important columns
    df["Lot Frontage"] = df.groupby("Neighborhood")["Lot Frontage"].transform(
        lambda x: x.fillna(x.median())
    )
    df["Lot Frontage"] = df["Lot Frontage"].fillna(df["Lot Frontage"].median())

    df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])

    # Categorial
    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    df[cat_cols] = df[cat_cols].fillna("None")

    # Numerical
    num_cols = df.select_dtypes(exclude=["object", "string"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    null_colls = df.isnull().sum()[df.isnull().sum() > 0]

    print("\nColumns with nulls:")
    print(null_colls.sort_values(ascending=False))
    print(f"Total: {null_colls.count()}")

    ################################
    # ENCODE CATEGORICAL VARIABLES #
    ################################

    df = pd.get_dummies(df, columns=cat_cols.tolist())

    print("\nData info after preprocessing:")
    print(df.info())

    ##############
    # SPLIT DATA #
    ##############

    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ##############
    # SCALE DATA #
    ##############

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    ###########################
    # PLOT CORRELATION MATRIX #
    ###########################

    corr = df.corr(numeric_only=True)

    target_corr = corr["SalePrice"].sort_values(ascending=False)

    top_features = target_corr.index[:15]

    plt.figure(figsize=(10, 8))
    sns.heatmap(df[top_features].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "correlation_matrix.png"))
    plt.show()
    
    ########################################
    # TRAIN AND EVALUATE LINEAR REGRESSION #
    ########################################
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred_lr = lr.predict(X_test)
    mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    print(f"\nLinear Regression:\n\tMAPE: {mape_lr*100:.4f}%\n\tR²: {r2_lr*100:.4f}%")

    #######################################
    # TRAIN AND EVALUATE RIDGE REGRESSION #
    #######################################

    ridge = Ridge(alpha=1, random_state=42)

    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)

    mape_ridge = mean_absolute_percentage_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)

    print(f"\nRidge:\n\tMAPE: {mape_ridge*100:.4f}%\n\tR²: {r2_ridge*100:.4f}%")
    
    ##################
    # COMPARE MODELS #
    ##################

    comparison = pd.DataFrame({
        'Model': ['Linear Regression', 'Ridge'],
        'MAPE': [mape_lr, mape_ridge],
        'R²': [r2_lr, r2_ridge]
    })
    print()
    print(comparison)


if __name__ == '__main__':
    main()
