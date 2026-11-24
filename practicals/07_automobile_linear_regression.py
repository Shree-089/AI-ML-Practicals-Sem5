"""
Practical 07: Linear Regression for automobile fuel-efficiency prediction.

Uses the UCI Auto MPG dataset. The target variable is mpg (miles per gallon).

Usage:
    python practicals/07_automobile_linear_regression.py
    python practicals/07_automobile_linear_regression.py path/to/auto-mpg.data
"""

import sys
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
)
COLUMN_NAMES = [
    "mpg", "cylinders", "displacement", "horsepower", "weight",
    "acceleration", "model_year", "origin", "car_name",
]
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "datasets" / "auto-mpg.data"


def load_auto_mpg():
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH

    if not data_path.exists():
        data_path.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading the UCI Auto MPG dataset...")
        urlretrieve(DATA_URL, data_path)

    if data_path.suffix.lower() == ".csv":
        dataframe = pd.read_csv(data_path)
    else:
        dataframe = pd.read_csv(
            data_path,
            sep=r"\s+",
            names=COLUMN_NAMES,
            na_values="?",
            quotechar='"',
            engine="python",
        )

    # Normalize common target-column spellings in user-provided CSV files.
    target_column = next(
        (column for column in dataframe.columns if column.strip().lower() in {"mpg", "miles_per_gallon"}),
        None,
    )
    if target_column is None:
        raise ValueError("Could not find the target column 'mpg' in the dataset.")

    dataframe = dataframe.rename(columns={target_column: "mpg"})
    dataframe["mpg"] = pd.to_numeric(dataframe["mpg"], errors="coerce")
    dataframe = dataframe.dropna(subset=["mpg"])

    # The car name is text and is not used as a predictor in this practical.
    features = dataframe.drop(columns=["mpg"])
    features = features.select_dtypes(include=["number"])
    if features.empty:
        raise ValueError("No numeric input features were found in the dataset.")

    return features, dataframe["mpg"]


def main():
    X, y = load_auto_mpg()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, X.columns.tolist())]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Linear Regression — Auto MPG")
    print("-----------------------------")
    print("Number of records:", len(X))
    print("Input features:", list(X.columns))
    print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
    print("Root Mean Squared Error:", mean_squared_error(y_test, predictions) ** 0.5)
    print("R-squared:", r2_score(y_test, predictions))

    coefficients = model.named_steps["regressor"].coef_
    coefficient_table = pd.DataFrame(
        {"Feature": X.columns, "Coefficient": coefficients}
    ).sort_values("Coefficient", key=abs, ascending=False)
    print("\nFeature coefficients (after standardization):")
    print(coefficient_table.to_string(index=False))

    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, predictions, alpha=0.75)
    lower = min(y_test.min(), predictions.min())
    upper = max(y_test.max(), predictions.max())
    plt.plot([lower, upper], [lower, upper], linestyle="--")
    plt.xlabel("Actual MPG")
    plt.ylabel("Predicted MPG")
    plt.title("Actual vs Predicted Fuel Efficiency")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
