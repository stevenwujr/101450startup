import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2


def forward_selection(X, y, max_features=5, test_size=0.2, random_state=42):
    X = pd.get_dummies(X, columns=['State'], drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    selected_features = []
    results = []

    for i in range(1, max_features + 1):
        best_feature = None
        best_rmse = np.inf
        best_r2 = -np.inf

        for feature in X_train.columns:
            if feature in selected_features:
                continue
            current_features = selected_features + [feature]
            model = LinearRegression()
            model.fit(X_train[current_features], y_train)
            pred = model.predict(X_test[current_features])
            rmse, r2 = evaluate(y_test, pred)

            if rmse < best_rmse:
                best_rmse = rmse
                best_r2 = r2
                best_feature = feature

        if best_feature is None:
            break

        selected_features.append(best_feature)
        results.append(
            {
                'Number of Features': i,
                'Selected Features': selected_features.copy(),
                'RMSE': best_rmse,
                'R-squared': best_r2,
            }
        )

    return pd.DataFrame(results)


def main():
    data = load_data('50_Startups.csv')
    X = data.drop(columns=['Profit'])
    y = data['Profit']

    results = forward_selection(X, y, max_features=5)
    print(results)


if __name__ == '__main__':
    main()
