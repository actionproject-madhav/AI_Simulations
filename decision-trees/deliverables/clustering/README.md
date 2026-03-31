# Iris Clustering Deliverable

## What this includes

- `clustering_analysis.py`: complete implementation for:
  - PCA scatter of true iris classes
  - K-Means clustering (`k=3`) and visualization
  - Petal-only scatter plots
  - Silhouette score plot for `k=2..10`
  - Hierarchical clustering dendrogram (Ward linkage)
  - Gaussian Mixture Model clustering (`k=3`)

- `plots/` (generated at runtime): all required `.png` output files.

## Run

From this directory:

```bash
python3 clustering_analysis.py
```

## Dependencies

Install if needed:

```bash
pip3 install scikit-learn matplotlib scipy numpy
```
