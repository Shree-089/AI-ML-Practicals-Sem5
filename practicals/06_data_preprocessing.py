"""
Practical 06: Basic data preprocessing.

Demonstrates:
1. Loading a CSV or creating a small example dataset.
2. Handling missing numerical and categorical values.
3. Encoding categorical features.
4. Normalizing numerical features.

Usage:
    python practicals/06_data_preprocessing.py
    python practicals/06_data_preprocessing.py path/to/data.csv
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data():
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        return pd.read_csv(csv_path)

    # Illustrative dataset containing missing numerical and categorical values.
    return pd.DataFrame(
        {
            "Age": [22, 25, None, 28, 24, 30],
            "Salary": [25000, 32000, 28000, None, 30000, 40000],
            "Department": ["IT", "HR", "IT", None, "Finance", "HR"],
            "City": ["Pune", "Mumbai", "Pune", "Pune", None, "Mumbai"],
        }
    )


def main():
    dataframe = load_data()

    print("Original dataset")
    print(dataframe)
    print("\nMissing values before preprocessing:")
    print(dataframe.isnull().sum())

    numerical_columns = dataframe.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = dataframe.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    transformers = []

    if numerical_columns:
        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numerical", numerical_pipeline, numerical_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    if not transformers:
        raise ValueError("The dataset must contain numerical or categorical columns.")

    preprocessor = ColumnTransformer(transformers=transformers)
    transformed = preprocessor.fit_transform(dataframe)

    feature_names = preprocessor.get_feature_names_out()
    processed_dataframe = pd.DataFrame(transformed, columns=feature_names)

    print("\nPreprocessed dataset")
    print(processed_dataframe)
    print("\nMissing values after preprocessing:")
    print(processed_dataframe.isnull().sum())
    print("\nFinal shape:", processed_dataframe.shape)


if __name__ == "__main__":
    main()
