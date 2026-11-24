"""
Practical 10: Decision Tree and Random Forest on the UCI Heart Disease data.

The Cleveland processed dataset is used. The original target `num` is
converted to a binary label: 0 = no disease, >0 = disease present.

Usage:
    python practicals/10_heart_disease_tree_forest.py
    python practicals/10_heart_disease_tree_forest.py path/to/heart.csv
"""

import sys
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
    "processed.cleveland.data"
)
COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "datasets" / "processed.cleveland.data"


def load_heart_data():
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH

    if not data_path.exists():
        data_path.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading the UCI Cleveland Heart Disease dataset...")
        urlretrieve(DATA_URL, data_path)

    if data_path.suffix.lower() == ".csv":
        dataframe = pd.read_csv(data_path, na_values="?")
    else:
        dataframe = pd.read_csv(
            data_path,
            names=COLUMN_NAMES,
            na_values="?",
            skipinitialspace=True,
        )

    dataframe.columns = [str(column).strip().lower() for column in dataframe.columns]
    target_name = "num" if "num" in dataframe.columns else "target"
    if target_name not in dataframe.columns:
        raise ValueError("Dataset must contain a 'num' or 'target' column.")

    dataframe = dataframe.replace("?", pd.NA)
    dataframe[target_name] = pd.to_numeric(dataframe[target_name], errors="coerce")
    dataframe = dataframe.dropna(subset=[target_name])

    y = (dataframe[target_name] > 0).astype(int)
    X = dataframe.drop(columns=[target_name])
    X = X.apply(pd.to_numeric, errors="coerce")

    # Remove columns that contain no usable numeric values.
    X = X.dropna(axis=1, how="all")
    if X.empty:
        raise ValueError("No usable numeric feature columns were found.")

    return X, y


def main():
    X, y = load_heart_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4, min_samples_split=4, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ),
    }

    fitted_models = {}
    print("Heart Disease Classification")
    print("----------------------------")
    print("Records:", len(X))
    print("Features:", list(X.columns))

    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        fitted_models[name] = pipeline

        print(f"\n{name}")
        print("Training accuracy:", pipeline.score(X_train, y_train))
        print("Testing accuracy:", accuracy_score(y_test, predictions))
        print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions, zero_division=0))

        cv_scores = cross_val_score(
            pipeline, X, y, cv=5, scoring="accuracy"
        )
        print("5-fold CV accuracy (mean):", cv_scores.mean())
        print("5-fold CV accuracy (std):", cv_scores.std())

    # Bias/variance discussion aid: compare a shallow and an unrestricted tree.
    print("\nDecision Tree complexity comparison")
    for depth in [1, 3, None]:
        tree_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", DecisionTreeClassifier(max_depth=depth, random_state=42)),
            ]
        )
        tree_pipeline.fit(X_train, y_train)
        train_score = tree_pipeline.score(X_train, y_train)
        test_score = tree_pipeline.score(X_test, y_test)
        print(
            f"max_depth={depth}: train accuracy={train_score:.3f}, "
            f"test accuracy={test_score:.3f}, gap={train_score - test_score:.3f}"
        )

    # Interpret Random Forest feature importance.
    forest = fitted_models["Random Forest"].named_steps["classifier"]
    importances = pd.Series(forest.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    print("\nRandom Forest feature importance:")
    print(importances.to_string())

    plt.figure(figsize=(8, 5))
    importances.sort_values().plot(kind="barh")
    plt.xlabel("Feature importance")
    plt.title("Random Forest — Feature Importance")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
