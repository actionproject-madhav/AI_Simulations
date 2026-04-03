# Clustering Writeup

## K-Means

Setosa is clearly separated in the PCA projection. Versicolor and virginica overlap, so k-means gets setosa perfect but misclassifies 11 versicolor and 14 virginica.

## Petal-Only

Setosa is still cleanly separable by petal measurements alone. Versicolor and virginica still overlap -- you can split one class out but not all three.

## Silhouette Plot

The silhouette score measures how well each point fits its cluster vs. the nearest other cluster: s = (b - a) / max(a, b). The best k here is 2 (score 0.58), not 3. That tells you the data has two natural groups statistically -- setosa vs. everything else -- even though there are three species. Cluster analysis finds structure in the data, not necessarily biological truth.

## Hierarchical Clustering

The dendrogram splits setosa off early at a large distance. Versicolor and virginica merge much later at a small distance, showing how similar they are.

## GMM

GMM actually does worse than k-means here. It gets setosa right but collapses almost all versicolor into the virginica component (48/50 wrong). The overlapping distributions cause EM to find a bad local solution. The fundamental problem is the same for all three methods -- versicolor and virginica are just too close in feature space to separate cleanly.
