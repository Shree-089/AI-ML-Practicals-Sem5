"""
Practical 11: K-Means clustering on a multidimensional dataset.

By default, this uses scikit-learn's Wine dataset (originally from UCI).
A local CSV can be supplied as the first command-line argument.

Usage:
    python practicals/11_kmeans_clustering.py
    python practicals/11_kmeans_clustering.py path/to/dataset.csv

The program scales numeric features, compares k values with silhouette
scores, selects the best score, and visualizes clusters using PCA.
"""

import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def load_features():
    if len(sys.argv) > 1:
        dataframe = pd.read_csv(sys.argv[1])
        features = dataframe.select_dtypes(include=["number"])
    else:
        dataset = load_wine(as_frame=True)
        features = dataset.data

    if features.empty:
        raise ValueError("The selected dataset has no numeric columns.")

    # Remove columns that are entirely missing, then impute remaining gaps.
    features = features.dropna(axis=1, how="all")
    if features.shape[1] == 0:
        raise ValueError("No numeric columns with usable values were found.")

    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(features)
    scaled = StandardScaler().fit_transform(imputed)
    return scaled, features.columns.tolist()


def main():
    X, feature_names = load_features()
    if len(X) < 3:
        raise ValueError("At least three records are required for clustering.")

    candidate_k = range(2, min(7, len(X)))
    scores = {}

    print("K-Means Clustering")
    print("------------------")
    print("Number of records:", len(X))
    print("Features:", feature_names)

    for k in candidate_k:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        scores[k] = score
        print(f"k={k}: silhouette score={score:.4f}")

    best_k = max(scores, key=scores.get)
    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = final_model.fit_predict(X)

    print("\nSelected number of clusters:", best_k)
    print("Best silhouette score:", scores[best_k])
    print("Cluster sizes:", pd.Series(labels).value_counts().sort_index().to_dict())

    reduced = PCA(n_components=2).fit_transform(X)

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.8,
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"K-Means Clusters (k={best_k}) — PCA View")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
