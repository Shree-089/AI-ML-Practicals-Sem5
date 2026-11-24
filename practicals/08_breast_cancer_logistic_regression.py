"""
Practical 08: Logistic Regression for Breast Cancer classification.

Uses scikit-learn's Breast Cancer Wisconsin Diagnostic dataset.
Reports accuracy, confusion matrix and ROC curve.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    dataset = load_breast_cancer()
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    matrix = confusion_matrix(y_test, predictions)
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    auc_value = roc_auc_score(y_test, probabilities)

    print("Logistic Regression — Breast Cancer")
    print("------------------------------------")
    print("Class labels:", dict(enumerate(dataset.target_names)))
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Confusion matrix:\n", matrix)
    print("ROC-AUC:", auc_value)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=dataset.target_names,
        yticklabels=dataset.target_names,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("True label")

    axes[1].plot(fpr, tpr, label=f"ROC curve (AUC = {auc_value:.3f})")
    axes[1].plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
