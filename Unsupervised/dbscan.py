# =============================================================================
# DBSCAN - Density-Based Spatial Clustering of Applications with Noise
# =============================================================================
# DBSCAN is an unsupervised clustering algorithm that groups points based on
# DENSITY rather than distance to a centroid (unlike K-Means).
#
# Core Idea: clusters are dense regions separated by sparse regions
#
# Key Parameters:
#   - eps (ε): the maximum distance between two points to be considered neighbors
#   - min_samples: minimum number of points to form a dense region (core point)
#
# Point Types:
#   - Core Point   : has at least min_samples neighbors within eps
#   - Border Point : within eps of a core point but has < min_samples neighbors
#   - Noise Point  : not within eps of any core point → labeled as -1 (outlier)
#
# Algorithm Steps:
#   1. Pick an unvisited point
#   2. If it's a core point, start a new cluster and expand it
#   3. Recursively add all density-reachable points to the cluster
#   4. If not a core point, mark as noise (may be revisited as border)
#   5. Repeat until all points are visited
#
# Advantages over K-Means:
#   ✓ No need to specify number of clusters K
#   ✓ Can find arbitrarily shaped clusters (not just spherical)
#   ✓ Robust to outliers (explicitly labels them as noise)
#   ✗ Sensitive to eps and min_samples
#   ✗ Struggles with varying density clusters
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

np.random.seed(42)

print("=" * 60)
print("DBSCAN - Density-Based Clustering")
print("=" * 60)

# =============================================================================
# PART A: Shapes K-Means CANNOT cluster (DBSCAN shines here)
# =============================================================================

# Generate non-spherical datasets
X_moons,   _ = make_moons(n_samples=300, noise=0.08, random_state=42)
X_circles, _ = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)
X_blobs,   _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)

datasets = [
    (X_moons,   "Two Moons"),
    (X_circles, "Concentric Circles"),
    (X_blobs,   "Gaussian Blobs"),
]

# DBSCAN parameters (tuned per dataset)
dbscan_params = [
    {'eps': 0.2,  'min_samples': 5},
    {'eps': 0.15, 'min_samples': 5},
    {'eps': 0.5,  'min_samples': 5},
]

# KMeans parameters (for comparison)
kmeans_params = [
    {'n_clusters': 2},
    {'n_clusters': 2},
    {'n_clusters': 3},
]

print("\n--- Comparing DBSCAN vs K-Means ---")

fig, axes = plt.subplots(3, 3, figsize=(15, 13))
fig.suptitle("DBSCAN vs K-Means on Different Cluster Shapes", fontsize=14, fontweight='bold')

for i, ((X, name), d_params, k_params) in enumerate(zip(datasets, dbscan_params, kmeans_params)):
    X_scaled = StandardScaler().fit_transform(X)

    # Fit DBSCAN
    db = DBSCAN(**d_params)
    db_labels = db.fit_predict(X_scaled)
    n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise       = (db_labels == -1).sum()

    # Fit K-Means
    km = KMeans(**k_params, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_scaled)

    print(f"\n{name}:")
    print(f"  DBSCAN  → {n_clusters_db} clusters, {n_noise} noise points")
    if n_clusters_db > 1:
        sil_db = silhouette_score(X_scaled, db_labels)
        print(f"  DBSCAN  Silhouette: {sil_db:.3f}")
    sil_km = silhouette_score(X_scaled, km_labels)
    print(f"  K-Means Silhouette: {sil_km:.3f}")

    # Column 0: Raw data
    axes[i, 0].scatter(X_scaled[:, 0], X_scaled[:, 1], c='gray', alpha=0.6, s=15)
    axes[i, 0].set_title(f"{name}\n(Raw Data)")

    # Column 1: DBSCAN result
    unique_labels = set(db_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))
    for j, label in enumerate(unique_labels):
        mask = db_labels == label
        color = 'black' if label == -1 else colors[j % len(colors)]
        marker = 'x' if label == -1 else 'o'
        lbl = 'Noise' if label == -1 else f'Cluster {label}'
        axes[i, 1].scatter(X_scaled[mask, 0], X_scaled[mask, 1],
                           c=[color], alpha=0.7, s=15, marker=marker, label=lbl)
    axes[i, 1].set_title(
        f"DBSCAN (eps={d_params['eps']}, min_samples={d_params['min_samples']})\n"
        f"{n_clusters_db} clusters, {n_noise} noise"
    )
    if n_clusters_db <= 5:
        axes[i, 1].legend(markerscale=2, fontsize=8)

    # Column 2: K-Means result
    scatter = axes[i, 2].scatter(X_scaled[:, 0], X_scaled[:, 1],
                                  c=km_labels, cmap='tab10', alpha=0.7, s=15)
    axes[i, 2].scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                       c='red', marker='X', s=150, zorder=5, label='Centroids')
    axes[i, 2].set_title(f"K-Means (K={k_params['n_clusters']})")
    axes[i, 2].legend(markerscale=1.5, fontsize=8)

plt.tight_layout()
plt.savefig("dbscan_shape_comparison.png", dpi=150)
plt.show()

# =============================================================================
# PART B: Hyperparameter Tuning with K-Distance Plot
# =============================================================================
print("\n--- K-Distance Plot for Optimal eps ---")

X_tune, _ = make_moons(n_samples=300, noise=0.08, random_state=42)
X_tune = StandardScaler().fit_transform(X_tune)

# K-distance: distance to k-th nearest neighbor for each point
# The "elbow" in this plot suggests a good eps value
nbrs = NearestNeighbors(n_neighbors=5).fit(X_tune)
distances, _ = nbrs.kneighbors(X_tune)
k_distances = np.sort(distances[:, -1])[::-1]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("DBSCAN Hyperparameter Analysis", fontsize=13, fontweight='bold')

axes[0].plot(k_distances, color='navy', linewidth=2)
axes[0].axhline(y=0.2, color='red', linestyle='--', label='Suggested eps ≈ 0.2')
axes[0].set_title("K-Distance Plot (k=5)\nElbow → Optimal eps")
axes[0].set_xlabel("Points sorted by distance")
axes[0].set_ylabel("5th Nearest Neighbor Distance")
axes[0].legend(); axes[0].grid(True)

# Effect of different eps values
eps_values = [0.05, 0.15, 0.2, 0.35]
colors_eps = ['red', 'blue', 'green', 'orange']
X_vis, _ = make_moons(n_samples=200, noise=0.08, random_state=42)
X_vis = StandardScaler().fit_transform(X_vis)

for k, (eps_val, col) in enumerate(zip(eps_values, colors_eps)):
    db_tmp = DBSCAN(eps=eps_val, min_samples=5).fit(X_vis)
    n_cl  = len(set(db_tmp.labels_)) - (1 if -1 in db_tmp.labels_ else 0)
    n_ns  = (db_tmp.labels_ == -1).sum()
    axes[1].scatter([], [], c=col, s=50, label=f"eps={eps_val}: {n_cl} clusters, {n_ns} noise")

# Show final with best eps
db_best = DBSCAN(eps=0.2, min_samples=5).fit(X_vis)
axes[1].scatter(X_vis[:, 0], X_vis[:, 1], c=db_best.labels_,
                cmap='tab10', alpha=0.7, s=20)
axes[1].set_title("Effect of eps on Clustering\n(showing best eps=0.2)")
axes[1].legend(fontsize=9); axes[1].grid(True)

plt.tight_layout()
plt.savefig("dbscan_results.png", dpi=150)
plt.show()
print("\nPlots saved as 'dbscan_shape_comparison.png' and 'dbscan_results.png'")
