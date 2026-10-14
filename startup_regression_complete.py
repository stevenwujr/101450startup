import gzip
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_URL = (
    "https://raw.githubusercontent.com/harimittapalli/"
    "Mulitple-Linear-Reggression/master/50_Startups.csv"
)


def load_data(csv_path: str | None = None) -> pd.DataFrame:
    """Load the startup dataset from a local file or URL."""
    if csv_path and Path(csv_path).exists():
        data = pd.read_csv(csv_path)
    else:
        try:
            data = pd.read_csv(DATA_URL)
        except Exception:
            raise FileNotFoundError(
                "Could not find 50_Startups.csv locally and could not download it from the URL."
            )

    print("Dataset preview:")
    print(data.head())
    print("\nColumns:", list(data.columns))
    print("Shape:", data.shape)
    return data


def prepare_data(data: pd.DataFrame, random_state: int = 42, test_size: float = 0.2):
    """Split into X/y and apply one-hot encoding for the State feature."""
    X = data.drop(columns=['Profit'])
    y = data['Profit']

    categorical_features = ['State']
    numerical_features = [col for col in X.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features),
        ],
        remainder='drop',
    )

    X_processed = preprocessor.fit_transform(X)
    feature_names = list(numerical_features) + list(
        preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test, feature_names


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("\nModel Evaluation")
    print(f"R-squared: {r2:.4f}")
    print(f"MAE:      {mae:.4f}")
    print(f"MSE:      {mse:.4f}")
    print(f"RMSE:     {rmse:.4f}")

    return {
        'y_pred': y_pred,
        'R2': r2,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
    }


def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='royalblue')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1.5)
    plt.title('Actual vs Predicted Profit')
    plt.xlabel('Actual Profit')
    plt.ylabel('Predicted Profit')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def forward_selection(X, y, max_features=5, test_size=0.2, random_state=42):
    """Simple forward selection using LinearRegression and RMSE as criterion."""
    X_df = pd.DataFrame(X)
    y_series = pd.Series(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=test_size, random_state=random_state
    )

    selected = []
    results = []

    for i in range(1, max_features + 1):
        best_feature = None
        best_rmse = np.inf
        best_r2 = -np.inf

        remaining = [col for col in X_train.columns if col not in selected]
        for feature in remaining:
            features_to_use = selected + [feature]
            model = LinearRegression()
            model.fit(X_train[features_to_use], y_train)
            y_pred = model.predict(X_test[features_to_use])
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            if rmse < best_rmse:
                best_feature = feature
                best_rmse = rmse
                best_r2 = r2

        if best_feature is None:
            break

        selected.append(best_feature)
        results.append(
            {
                'Number of Features': i,
                'Selected Features': selected.copy(),
                'RMSE': best_rmse,
                'R-squared': best_r2,
            }
        )

    return pd.DataFrame(results)


def save_model(model, filepath: str):
    """Save a trained model as a gzip-compressed pickle file for deployment."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved to {path}")


def load_model(filepath: str):
    """Load a serialized model from a gzip-compressed pickle file."""
    with gzip.open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model


def main():
    print("=== CRISP-DM: Business Understanding ===")
    print("Objective: Predict startup profit using R&D, Administration, Marketing, and State.")

    data = load_data('50_Startups.csv')

    print("\n=== CRISP-DM: Data Preparation ===")
    X_train, X_test, y_train, y_test, feature_names = prepare_data(data)
    print("Feature names after preprocessing:")
    print(feature_names)

    print("\n=== CRISP-DM: Modeling ===")
    model = train_linear_regression(X_train, y_train)
    print("Model trained.")

    print("\n=== CRISP-DM: Evaluation ===")
    metrics = evaluate_model(model, X_test, y_test)
    plot_actual_vs_predicted(y_test, metrics['y_pred'])

    print("\n=== Feature Selection Demo ===")
    selection_df = forward_selection(
        X_train,
        y_train,
        max_features=5,
        test_size=0.2,
        random_state=42,
    )
    print(selection_df)

    print("\n=== Deployment ===")
    save_model(model, 'artifacts/startup_model.pkl.gz')

    loaded_model = load_model('artifacts/startup_model.pkl.gz')
    print("Model load check successful:", type(loaded_model).__name__)

    print("\n=== Summary ===")
    print("The model may be used to estimate startup profitability based on the provided predictors.")
    print("This workflow follows the CRISP-DM cycle: Business Understanding → Data Preparation → Modeling → Evaluation → Deployment.")


if __name__ == '__main__':
    main()
