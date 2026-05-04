# =============================================================================
# K-MEANS CLUSTERING
# =============================================================================
# K-Means is an UNSUPERVISED learning algorithm that groups data into K clusters
# by iteratively assigning points to the nearest centroid and updating centroids.
#
# Algorithm Steps:
#   1. Initialize: randomly place K centroids
#   2. Assignment: assign each point to the nearest centroid
#      (using Euclidean distance: d = sqrt(Σ(xi - ci)²))
#   3. Update: move each centroid to the mean of its assigned points
#   4. Repeat steps 2-3 until convergence (centroids stop moving)
#
# Key Concepts:
#   - Centroid: the mean position of all points in a cluster
#   - Inertia (WCSS): Within-Cluster Sum of Squares — measures cluster compactness
#     WCSS = Σ Σ ||x - μk||²   (lower = better, tighter clusters)
#   - Elbow Method: plot WCSS vs K, look for the "elbow" to find optimal K
#   - Silhouette Score: measures how similar a point is to its own cluster
#     vs other clusters. Range [-1, 1], higher is better.
#
# Limitations:
#   - Must specify K in advance
#   - Sensitive to initial centroid placement (use k-means++ initialization)
#   - Assumes spherical clusters of similar size
#   - Sensitive to outliers
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("K-MEANS CLUSTERING")
print("=" * 60)

# =============================================================================
# PART A: SYNTHETIC DATA — Understanding K-Means
# =============================================================================
print("\n--- Part A: Synthetic Blob Data ---")

# Generate clear clusters for visualization
X_blobs, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.8,
                              random_state=42)

scaler = StandardScaler()
X_blobs_scaled = scaler.fit_transform(X_blobs)

# --- Elbow Method: Find optimal K ---
print("\nRunning Elbow Method (K = 1 to 10)...")
wcss       = []
sil_scores = []
K_range    = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X_blobs_scaled)
    wcss.append(km.inertia_)
    sil_scores.append(silhouette_score(X_blobs_scaled, km.labels_))
    print(f"  K={k}: WCSS={km.inertia_:.1f}, Silhouette={sil_scores[-1]:.3f}")

# Fit with optimal K=4
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
kmeans.fit(X_blobs_scaled)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

ari = adjusted_rand_score(y_true, labels)
sil = silhouette_score(X_blobs_scaled, labels)

print(f"\nFinal Model (K=4):")
print(f"  Inertia (WCSS)   : {kmeans.inertia_:.2f}")
print(f"  Silhouette Score : {sil:.4f}  (1.0 = perfect)")
print(f"  Adjusted Rand    : {ari:.4f}  (1.0 = matches true labels)")
print(f"  Iterations       : {kmeans.n_iter_}")

# =============================================================================
# PART B: IRIS DATASET — Real-world Application
# =============================================================================
print("\n--- Part B: Iris Dataset ---")

iris = load_iris()
X_iris = StandardScaler().fit_transform(iris.data)

kmeans_iris = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
kmeans_iris.fit(X_iris)

ari_iris = adjusted_rand_score(iris.target, kmeans_iris.labels_)
sil_iris = silhouette_score(X_iris, kmeans_iris.labels_)
print(f"  Silhouette Score : {sil_iris:.4f}")
print(f"  Adjusted Rand    : {ari_iris:.4f}")

# =============================================================================
# VISUALIZATION
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("K-Means Clustering", fontsize=15, fontweight='bold')

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

# 1. Raw Data
axes[0, 0].scatter(X_blobs_scaled[:, 0], X_blobs_scaled[:, 1],
                   c='gray', alpha=0.5, s=20)
axes[0, 0].set_title("1. Raw Data (No Labels)")
axes[0, 0].set_xlabel("Feature 1"); axes[0, 0].set_ylabel("Feature 2")

# 2. K-Means Result
for i in range(4):
    mask = labels == i
    axes[0, 1].scatter(X_blobs_scaled[mask, 0], X_blobs_scaled[mask, 1],
                       c=colors[i], alpha=0.6, s=20, label=f'Cluster {i+1}')
axes[0, 1].scatter(centers[:, 0], centers[:, 1],
                   c='black', marker='X', s=200, zorder=5, label='Centroids')
axes[0, 1].set_title(f"2. K-Means Result (K=4)\nSilhouette={sil:.3f}, ARI={ari:.3f}")
axes[0, 1].legend(); axes[0, 1].set_xlabel("Feature 1")

# 3. True Labels
for i in range(4):
    mask = y_true == i
    axes[0, 2].scatter(X_blobs_scaled[mask, 0], X_blobs_scaled[mask, 1],
                       c=colors[i], alpha=0.6, s=20, label=f'True {i+1}')
axes[0, 2].set_title("3. True Labels (for comparison)")
axes[0, 2].legend(); axes[0, 2].set_xlabel("Feature 1")

# 4. Elbow Method
axes[1, 0].plot(list(K_range), wcss, 'bо-', linewidth=2, markersize=8)
axes[1, 0].set_title("4. Elbow Method (find the 'elbow')")
axes[1, 0].set_xlabel("Number of Clusters (K)")
axes[1, 0].set_ylabel("WCSS (Inertia)")
axes[1, 0].axvline(x=4, color='red', linestyle='--', label='Optimal K=4')
axes[1, 0].legend(); axes[1, 0].grid(True)

# 5. Silhouette Scores
axes[1, 1].bar(list(K_range), sil_scores, color='steelblue', alpha=0.8)
axes[1, 1].set_title("5. Silhouette Score vs K")
axes[1, 1].set_xlabel("Number of Clusters (K)")
axes[1, 1].set_ylabel("Silhouette Score")
axes[1, 1].axvline(x=4, color='red', linestyle='--', label='Optimal K=4')
axes[1, 1].legend(); axes[1, 1].grid(True, axis='y')

# 6. Iris with PCA
pca = PCA(n_components=2)
X_iris_2d = pca.fit_transform(X_iris)
iris_colors = ['#e41a1c', '#377eb8', '#4daf4a']
for i in range(3):
    mask = kmeans_iris.labels_ == i
    axes[1, 2].scatter(X_iris_2d[mask, 0], X_iris_2d[mask, 1],
                       c=iris_colors[i], alpha=0.7, s=30,
                       label=f'Cluster {i+1}')
centers_2d = pca.transform(kmeans_iris.cluster_centers_)
axes[1, 2].scatter(centers_2d[:, 0], centers_2d[:, 1],
                   c='black', marker='X', s=200, zorder=5, label='Centroids')
axes[1, 2].set_title(f"6. K-Means on Iris (PCA 2D)\nSilhouette={sil_iris:.3f}")
axes[1, 2].legend(); axes[1, 2].set_xlabel("PC1"); axes[1, 2].set_ylabel("PC2")

plt.tight_layout()
plt.savefig("kmeans_results.png", dpi=150)
plt.show()
print("\nPlot saved as 'kmeans_results.png'")
