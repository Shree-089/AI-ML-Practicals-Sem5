"""
Practical 12: Hierarchical clustering on the Iris dataset.

Creates a dendrogram, fits Agglomerative (hierarchical) clustering,
fits K-Means for comparison, and reports silhouette scores.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def main():
    dataset = load_iris()
    X = StandardScaler().fit_transform(dataset.data)
    number_of_clusters = 3

    # Ward linkage merges clusters to minimize within-cluster variance.
    linkage_matrix = linkage(X, method="ward")

    hierarchical = AgglomerativeClustering(
        n_clusters=number_of_clusters,
        linkage="ward",
    )
    hierarchical_labels = hierarchical.fit_predict(X)

    kmeans = KMeans(
        n_clusters=number_of_clusters,
        random_state=42,
        n_init=10,
    )
    kmeans_labels = kmeans.fit_predict(X)

    hierarchical_score = silhouette_score(X, hierarchical_labels)
    kmeans_score = silhouette_score(X, kmeans_labels)

    print("Hierarchical Clustering — Iris Dataset")
    print("---------------------------------------")
    print("Number of clusters:", number_of_clusters)
    print("Hierarchical silhouette score:", hierarchical_score)
    print("K-Means silhouette score:", kmeans_score)

    # Dendrogram visualization.
    plt.figure(figsize=(12, 5))
    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=90,
        leaf_font_size=8,
    )
    plt.title("Iris Hierarchical Clustering Dendrogram (Ward Linkage)")
    plt.xlabel("Sample / merged cluster")
    plt.ylabel("Ward distance")
    plt.tight_layout()

    # Project the original four features into two dimensions for plotting.
    reduced = PCA(n_components=2).fit_transform(X)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.scatterplot(
        x=reduced[:, 0],
        y=reduced[:, 1],
        hue=hierarchical_labels,
        palette="Set1",
        ax=axes[0],
        legend="full",
    )
    axes[0].set_title("Agglomerative Clustering (PCA View)")
    axes[0].set_xlabel("Principal Component 1")
    axes[0].set_ylabel("Principal Component 2")

    sns.scatterplot(
        x=reduced[:, 0],
        y=reduced[:, 1],
        hue=kmeans_labels,
        palette="Set2",
        ax=axes[1],
        legend="full",
    )
    axes[1].set_title("K-Means Clustering (PCA View)")
    axes[1].set_xlabel("Principal Component 1")
    axes[1].set_ylabel("Principal Component 2")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
