"""
Practical 09: Bayesian classification on the Iris dataset.

Gaussian Naive Bayes is trained and evaluated using accuracy,
confusion matrix, precision and recall.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def main():
    dataset = load_iris()
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    matrix = confusion_matrix(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="macro", zero_division=0)
    recall = recall_score(y_test, predictions, average="macro", zero_division=0)

    print("Gaussian Naive Bayes — Iris Dataset")
    print("-----------------------------------")
    print("Accuracy:", accuracy)
    print("Macro precision:", precision)
    print("Macro recall:", recall)
    print("\nConfusion matrix:\n", matrix)
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=dataset.target_names,
            zero_division=0,
        )
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=dataset.target_names,
        yticklabels=dataset.target_names,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Iris — Gaussian Naive Bayes Confusion Matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
