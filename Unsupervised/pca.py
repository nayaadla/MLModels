# =============================================================================
# PRINCIPAL COMPONENT ANALYSIS (PCA)
# =============================================================================
# PCA is an unsupervised dimensionality reduction technique that transforms
# high-dimensional data into a lower-dimensional space while preserving
# maximum variance.
#
# Core Idea: Find the directions (principal components) of maximum variance
# in the data, then project the data onto those directions.
#
# Mathematical Steps:
#   1. Standardize the data (zero mean, unit variance)
#   2. Compute the Covariance Matrix: Σ = (1/n) * Xᵀ * X
#   3. Compute Eigenvalues & Eigenvectors of Σ
#      - Eigenvectors = principal component directions
#      - Eigenvalues  = amount of variance explained by each component
#   4. Sort eigenvectors by descending eigenvalue
#   5. Select top K eigenvectors (principal components)
#   6. Project data: Z = X · W  (W = selected eigenvectors)
#
# Key Concepts:
#   - Principal Components (PCs): orthogonal directions of max variance
#   - Explained Variance Ratio: how much variance each PC captures
#   - Cumulative Explained Variance: how many PCs to keep (e.g., 95% threshold)
#   - Reconstruction: can approximate original data from reduced representation
#
# Applications:
#   - Visualization: reduce to 2D/3D for plotting
#   - Noise reduction: low-variance components often contain noise
#   - Speed up ML: fewer features = faster training
#   - Feature extraction: uncorrelated principal components
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns

np.random.seed(42)

print("=" * 60)
print("PCA - Principal Component Analysis")
print("=" * 60)

# =============================================================================
# PART A: Understanding PCA on Digits Dataset (64D → 2D visualization)
# =============================================================================
print("\n--- Part A: Digits Dataset (64 → 2D) ---")

digits = load_digits()
X      = digits.data          # 1797 samples, 64 features
y      = digits.target

scaler  = StandardScaler()
X_std   = scaler.fit_transform(X)

# Fit PCA with ALL components to analyze variance
pca_full = PCA().fit(X_std)

# Explained variance
exp_var     = pca_full.explained_variance_ratio_
cum_exp_var = np.cumsum(exp_var)

# How many components for 95% variance?
n_95 = np.argmax(cum_exp_var >= 0.95) + 1
n_99 = np.argmax(cum_exp_var >= 0.99) + 1
print(f"Original dimensions     : {X.shape[1]}")
print(f"Components for 95% var  : {n_95}")
print(f"Components for 99% var  : {n_99}")
print(f"Top 5 PCs explain       : {cum_exp_var[4]*100:.1f}% of variance")
print(f"Top 10 PCs explain      : {cum_exp_var[9]*100:.1f}% of variance")

# 2D projection for visualization
pca_2d = PCA(n_components=2)
X_2d   = pca_2d.fit_transform(X_std)

print(f"\n2 PCs explain: {pca_2d.explained_variance_ratio_.sum()*100:.1f}% of variance")

# =============================================================================
# PART B: PCA for Dimensionality Reduction + ML
# =============================================================================
print("\n--- Part B: PCA + Classification Speed Comparison ---")

cancer = load_breast_cancer()
X_c    = StandardScaler().fit_transform(cancer.data)

results = {}
for n_comp in [2, 5, 10, 15, 20, 30]:
    pca_tmp = PCA(n_components=n_comp)
    X_reduced = pca_tmp.fit_transform(X_c)
    var_exp   = pca_tmp.explained_variance_ratio_.sum()

    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, cancer.target, test_size=0.2, random_state=42
    )
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    results[n_comp] = {'acc': acc, 'var': var_exp}
    print(f"  n_components={n_comp:2d}: Acc={acc*100:.1f}%, Var={var_exp*100:.1f}%")

# Full features baseline
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_c, cancer.target, test_size=0.2, random_state=42
)
clf_full = LogisticRegression(max_iter=1000, random_state=42)
clf_full.fit(X_train_f, y_train_f)
acc_full = accuracy_score(y_test_f, clf_full.predict(X_test_f))
print(f"  Full (30 feat):  Acc={acc_full*100:.1f}%, Var=100.0%")

# =============================================================================
# PART C: Image Reconstruction with PCA
# =============================================================================
print("\n--- Part C: Image Reconstruction ---")

face_idx = 0
original = X_std[face_idx]

recon_errors = []
n_comp_range = [1, 5, 10, 20, 30, 40, 64]
for n in n_comp_range:
    pca_r    = PCA(n_components=n).fit(X_std)
    X_r      = pca_r.transform(X_std)
    X_recon  = pca_r.inverse_transform(X_r)
    err      = np.mean((X_std - X_recon) ** 2)
    recon_errors.append(err)

# =============================================================================
# VISUALIZATION
# =============================================================================
fig = plt.figure(figsize=(20, 14))
fig.suptitle("PCA - Principal Component Analysis", fontsize=15, fontweight='bold')

# 1. Explained Variance
ax1 = fig.add_subplot(3, 3, 1)
ax1.bar(range(1, 21), exp_var[:20], alpha=0.8, color='steelblue', label='Individual')
ax1.step(range(1, 21), cum_exp_var[:20], where='mid', color='red', lw=2, label='Cumulative')
ax1.axhline(0.95, color='green', linestyle='--', label='95% threshold')
ax1.set_title("1. Explained Variance per Component")
ax1.set_xlabel("Principal Component"); ax1.set_ylabel("Explained Variance Ratio")
ax1.legend(); ax1.grid(True, alpha=0.3)

# 2. Cumulative Explained Variance
ax2 = fig.add_subplot(3, 3, 2)
ax2.plot(range(1, len(cum_exp_var)+1), cum_exp_var * 100, 'b-', lw=2)
ax2.axhline(95, color='red',   linestyle='--', label=f'95% → {n_95} PCs')
ax2.axhline(99, color='green', linestyle='--', label=f'99% → {n_99} PCs')
ax2.set_title("2. Cumulative Explained Variance")
ax2.set_xlabel("Number of Components"); ax2.set_ylabel("Cumulative Variance (%)")
ax2.legend(); ax2.grid(True, alpha=0.3)

# 3. 2D Projection
ax3 = fig.add_subplot(3, 3, 3)
scatter = ax3.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', alpha=0.6, s=15)
plt.colorbar(scatter, ax=ax3, label='Digit')
ax3.set_title(f"3. Digits in 2D (PCA)\n{pca_2d.explained_variance_ratio_.sum()*100:.1f}% variance")
ax3.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
ax3.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")

# 4. First 6 Principal Components (what they look like as images)
ax_pc = [fig.add_subplot(3, 3, i) for i in [4, 5, 6]]
for i, ax in enumerate(ax_pc):
    pc_img = pca_full.components_[i].reshape(8, 8)
    im = ax.imshow(pc_img, cmap='RdBu_r', interpolation='nearest')
    ax.set_title(f"PC{i+1} ({exp_var[i]*100:.1f}% var)")
    ax.axis('off')
    plt.colorbar(im, ax=ax)

# 5. Image Reconstruction at different n_components
pca_recon = PCA(n_components=64).fit(X_std)
ax_orig = fig.add_subplot(3, 3, 7)
ax_orig.imshow(X_std[face_idx].reshape(8, 8), cmap='gray')
ax_orig.set_title("Original (64D)")
ax_orig.axis('off')

for i, n in enumerate([10, 30]):
    pca_n   = PCA(n_components=n).fit(X_std)
    x_recon = pca_n.inverse_transform(pca_n.transform(X_std[[face_idx]]))
    ax_r    = fig.add_subplot(3, 3, 8 + i)
    ax_r.imshow(x_recon.reshape(8, 8), cmap='gray')
    var_n   = pca_n.explained_variance_ratio_.sum()
    ax_r.set_title(f"Reconstructed ({n} PCs)\n{var_n*100:.1f}% var")
    ax_r.axis('off')

plt.tight_layout()
plt.savefig("pca_results.png", dpi=150)
plt.show()
print("\nPlot saved as 'pca_results.png'")
