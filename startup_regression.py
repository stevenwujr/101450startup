import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def load_data(csv_path: str) -> pd.DataFrame:
    """Load startup dataset from local CSV file."""
    data = pd.read_csv(csv_path)
    print("Dataset loaded successfully.")
    print(data.head())
    print("\nColumns:", list(data.columns))
    print("\nShape:", data.shape)
    return data


def prepare_data(data: pd.DataFrame):
    """Prepare features and target for multiple linear regression."""
    # CRISP-DM: Data Preparation
    X = data.drop(columns=['Profit'])
    y = data['Profit']

    # One-hot encoding for the categorical feature 'State'
    # 'drop_first=True' helps avoid the dummy variable trap
    X = pd.get_dummies(X, columns=['State'], drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Train the multiple linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Predict on the test set and compute evaluation metrics."""
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Evaluation")
    print(f"R-squared: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return y_pred, {'R2': r2, 'MAE': mae, 'MSE': mse, 'RMSE': rmse}


def plot_predictions(y_test, y_pred):
    """Create actual vs predicted profit scatter plot."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.title('Actual vs Predicted Profit')
    plt.xlabel('Actual Profit')
    plt.ylabel('Predicted Profit')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    # CRISP-DM Step 1: Business Understanding
    # Goal: predict startup profit using multiple variables such as R&D spend,
    # administration, marketing spend, and state.

    # Step 1: Data Acquisition
    data = load_data('50_Startups.csv')

    # Step 2: Data Preparation
    X_train, X_test, y_train, y_test = prepare_data(data)

    # Step 3: Modeling
    model = train_model(X_train, y_train)

    # Step 4: Evaluation
    y_pred, metrics = evaluate_model(model, X_test, y_test)

    # Optional visualization
    plot_predictions(y_test, y_pred)

    # Print coefficients for reference
    print("\nModel coefficients:")
    feature_names = list(X_train.columns)
    for name, coef in zip(feature_names, model.coef_):
        print(f"{name}: {coef:.6f}")
    print(f"Intercept: {model.intercept_:.6f}")

    print("\nCompleted: Multiple Linear Regression on Startup Profit data.")


if __name__ == '__main__':
    main()
