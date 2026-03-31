"""
Clustering analysis deliverable on Fisher's Iris dataset.

Generates only the required deliverable charts:
- PCA scatter with true species labels
- PCA scatter with K-Means (k=3) assignments
- Petal-only scatter (true labels)
- Silhouette score curve for k in [2, 10]
- Hierarchical clustering dendrogram (Ward linkage)
- GMM clustering scatter (k=3)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path(__file__).resolve().parent / "plots"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close()


def remap_clusters_to_labels(cluster_ids: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
    """
    Map arbitrary cluster IDs to best-matching true class labels.
    This is only for interpretability when comparing to known iris species.
    """
    remapped = np.zeros_like(cluster_ids)
    for cluster_id in np.unique(cluster_ids):
        mask = cluster_ids == cluster_id
        majority_label = np.bincount(true_labels[mask]).argmax()
        remapped[mask] = majority_label
    return remapped


def plot_true_labels_pca(X_scaled: np.ndarray, y: np.ndarray, target_names: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    for class_id, class_name in enumerate(target_names):
        pts = X_pca[y == class_id]
        plt.scatter(pts[:, 0], pts[:, 1], s=45, alpha=0.85, label=class_name)

    plt.title("Iris: True Species (PCA projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    save_figure("01_pca_true_labels.png")
    return X_pca


def run_kmeans_and_plot(X_scaled: np.ndarray, X_pca: np.ndarray, y: np.ndarray, target_names: np.ndarray) -> None:
    kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    mapped = remap_clusters_to_labels(kmeans_labels, y)

    plt.figure(figsize=(8, 6))
    for class_id, class_name in enumerate(target_names):
        pts = X_pca[mapped == class_id]
        plt.scatter(pts[:, 0], pts[:, 1], s=45, alpha=0.85, label=f"Cluster -> {class_name}")
    plt.title("K-Means (k=3) Cluster Assignments on PCA space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    save_figure("02_pca_kmeans_k3.png")

    cm = confusion_matrix(y, mapped)
    print("\nK-Means (k=3) confusion matrix (rows=true, cols=pred):")
    print(cm)


def plot_petal_only(X: np.ndarray, y: np.ndarray, target_names: np.ndarray) -> None:
    # Features in sklearn iris: [sepal_len, sepal_wid, petal_len, petal_wid]
    petal_len = X[:, 2]
    petal_wid = X[:, 3]
    X_petal = np.column_stack([petal_len, petal_wid])

    # True labels
    plt.figure(figsize=(8, 6))
    for class_id, class_name in enumerate(target_names):
        pts = X_petal[y == class_id]
        plt.scatter(pts[:, 0], pts[:, 1], s=45, alpha=0.85, label=class_name)
    plt.title("Iris using only Petal Length & Petal Width (True Labels)")
    plt.xlabel("Petal length")
    plt.ylabel("Petal width")
    plt.legend()
    save_figure("03_petal_only_true_labels.png")

def plot_silhouette_curve(X_scaled: np.ndarray) -> int:
    ks = list(range(2, 11))
    scores = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(score)

    best_idx = int(np.argmax(scores))
    best_k = ks[best_idx]

    plt.figure(figsize=(8, 5.5))
    plt.plot(ks, scores, marker="o")
    plt.title("Silhouette Scores for K-Means on Iris")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Average silhouette score")
    plt.xticks(ks)
    plt.grid(alpha=0.25)
    plt.axvline(best_k, linestyle="--", alpha=0.7, label=f"Best k={best_k}")
    plt.legend()
    save_figure("04_silhouette_k2_to_k10.png")

    print("\nSilhouette scores:")
    for k, score in zip(ks, scores):
        print(f"k={k}: {score:.4f}")
    print(f"Best k by silhouette: {best_k}")
    return best_k


def plot_hierarchical_dendrogram(X_scaled: np.ndarray) -> None:
    # Ward linkage minimizes increase in within-cluster variance.
    Z = linkage(X_scaled, method="ward")
    plt.figure(figsize=(12, 6))
    dendrogram(Z, leaf_rotation=90, leaf_font_size=6, no_labels=True)
    plt.title("Hierarchical Clustering Dendrogram (Ward linkage)")
    plt.xlabel("Samples")
    plt.ylabel("Linkage distance")
    save_figure("05_dendrogram_ward.png")


def run_gmm_and_plot(X_scaled: np.ndarray, X_pca: np.ndarray, y: np.ndarray, target_names: np.ndarray) -> None:
    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
    gmm_labels = gmm.fit_predict(X_scaled)
    mapped = remap_clusters_to_labels(gmm_labels, y)

    plt.figure(figsize=(8, 6))
    for class_id, class_name in enumerate(target_names):
        pts = X_pca[mapped == class_id]
        plt.scatter(pts[:, 0], pts[:, 1], s=45, alpha=0.85, label=f"GMM -> {class_name}")
    plt.title("Gaussian Mixture Model (k=3) on PCA space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    save_figure("06_pca_gmm_k3.png")

    cm = confusion_matrix(y, mapped)
    print("\nGMM (k=3) confusion matrix (rows=true, cols=pred):")
    print(cm)


def print_short_answers(best_k: int) -> None:
    print("\n" + "=" * 68)
    print("Short analysis answers")
    print("=" * 68)
    print(
        "- Petal-only features clearly separate setosa from the other two classes,\n"
        "  but versicolor and virginica still overlap."
    )
    print(
        "- Silhouette score uses each sample's mean intra-cluster distance (a)\n"
        "  and nearest-cluster mean distance (b), with s = (b-a)/max(a,b).\n"
        "  The curve's peak indicates the best k under this criterion."
    )
    print(
        f"- Best k on this run is {best_k}, which suggests the 'natural' cluster\n"
        "  count under compact-separation assumptions may differ from biological classes."
    )
    print(
        "- Cluster analysis challenge here: versicolor and virginica form partially\n"
        "  overlapping distributions in feature space, so crisp boundaries are imperfect."
    )


def main() -> None:
    ensure_output_dir()

    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_pca = plot_true_labels_pca(X_scaled, y, target_names)
    run_kmeans_and_plot(X_scaled, X_pca, y, target_names)
    plot_petal_only(X, y, target_names)
    best_k = plot_silhouette_curve(X_scaled)
    plot_hierarchical_dendrogram(X_scaled)
    run_gmm_and_plot(X_scaled, X_pca, y, target_names)
    print_short_answers(best_k)

    print(f"\nSaved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
