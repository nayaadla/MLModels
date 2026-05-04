# =============================================================================
# LABEL PROPAGATION - Semi-Supervised Learning
# =============================================================================
# Semi-supervised learning uses a SMALL amount of labeled data combined with
# a LARGE amount of unlabeled data for training. This is realistic because:
#   - Labeling data is expensive and time-consuming
#   - Collecting unlabeled data is cheap
#
# Label Propagation works on a GRAPH:
#   - Each data point is a node
#   - Edges connect similar points (based on distance)
#   - Edge weights = similarity between points
#   - Labels "flow" from labeled nodes to unlabeled nodes through edges
#
# Algorithm Steps:
#   1. Build a similarity graph (kernel matrix W)
#      RBF kernel: W_ij = exp(-γ * ||xi - xj||²)
#      KNN kernel: connect each point to its K nearest neighbors
#   2. Construct transition matrix T = D⁻¹W (row-normalized)
#      D = diagonal degree matrix (Dii = Σj Wij)
#   3. Propagate labels: F(t+1) = T * F(t)
#      F = matrix of class probability distributions
#   4. Clamp labeled points back to their true labels after each step
#   5. Repeat until convergence
#
# Key Insight: Points close to each other in feature space likely share labels.
# The algorithm respects the "cluster assumption": decision boundaries lie
# in low-density regions.
#
# Parameters:
#   - kernel: 'rbf' (smooth, uses all neighbors) or 'knn' (sparse graph)
#   - gamma: controls width of RBF kernel (higher = more local influence)
#   - n_neighbors: used with knn kernel
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.datasets import load_digits, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import seaborn as sns

np.random.seed(42)

print("=" * 60)
print("LABEL PROPAGATION - Semi-Supervised Learning")
print("=" * 60)

# =============================================================================
# PART A: Visual Intuition with 2D Data (Two Moons)
# =============================================================================
print("\n--- Part A: Visual Demo (Two Moons) ---")

X_moons, y_moons = make_moons(n_samples=200, noise=0.1, random_state=42)
X_moons = StandardScaler().fit_transform(X_moons)

# Simulate having only 10% labeled data
n_labeled  = 10
n_total    = len(y_moons)
labeled_idx = np.random.choice(n_total, n_labeled, replace=False)

# Create y_partial: keep labels only for labeled_idx, set rest to -1
# -1 is the sklearn convention for "unlabeled"
y_partial = np.full(n_total, -1)
y_partial[labeled_idx] = y_moons[labeled_idx]

print(f"Total samples    : {n_total}")
print(f"Labeled samples  : {n_labeled} ({n_labeled/n_total*100:.0f}%)")
print(f"Unlabeled samples: {n_total - n_labeled} ({(n_total-n_labeled)/n_total*100:.0f}%)")

# Train Label Propagation
lp_moons = LabelPropagation(kernel='rbf', gamma=10, max_iter=1000)
lp_moons.fit(X_moons, y_partial)
y_pred_moons = lp_moons.predict(X_moons)

# Compare with supervised-only baseline (trained on labeled only)
X_lab  = X_moons[labeled_idx]
y_lab  = y_moons[labeled_idx]
lr_base = LogisticRegression()
lr_base.fit(X_lab, y_lab)
y_pred_base = lr_base.predict(X_moons)

acc_lp   = accuracy_score(y_moons, y_pred_moons)
acc_base = accuracy_score(y_moons, y_pred_base)
print(f"\nLabel Propagation accuracy (all): {acc_lp*100:.1f}%")
print(f"Supervised-only accuracy (all) : {acc_base*100:.1f}%")

# =============================================================================
# PART B: Digits Dataset — Varying Label Percentage
# =============================================================================
print("\n--- Part B: Digits Dataset (varying label %) ---")

digits = load_digits()
X_d    = StandardScaler().fit_transform(digits.data)
y_d    = digits.target

# Test with different amounts of labeled data
label_fractions  = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
acc_lp_list      = []
acc_spread_list  = []
acc_supervised   = []

for frac in label_fractions:
    n_lab = max(int(frac * len(y_d)), 10)

    # Randomly select labeled points (stratified)
    lab_idx = []
    for cls in np.unique(y_d):
        cls_idx  = np.where(y_d == cls)[0]
        n_select = max(1, int(frac * len(cls_idx)))
        lab_idx.extend(np.random.choice(cls_idx, n_select, replace=False))
    lab_idx = np.array(lab_idx)

    y_partial_d = np.full(len(y_d), -1)
    y_partial_d[lab_idx] = y_d[lab_idx]

    # Label Propagation
    lp = LabelPropagation(kernel='knn', n_neighbors=7, max_iter=1000)
    lp.fit(X_d, y_partial_d)
    acc_lp_list.append(accuracy_score(y_d, lp.predict(X_d)))

    # Label Spreading (similar but uses normalized graph Laplacian)
    ls = LabelSpreading(kernel='knn', n_neighbors=7, alpha=0.2, max_iter=1000)
    ls.fit(X_d, y_partial_d)
    acc_spread_list.append(accuracy_score(y_d, ls.predict(X_d)))

    # Supervised only (trained on labeled subset)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_d[lab_idx], y_d[lab_idx])
    acc_supervised.append(accuracy_score(y_d, lr.predict(X_d)))

    print(f"  {frac*100:.0f}% labeled ({len(lab_idx)} pts): "
          f"LP={acc_lp_list[-1]*100:.1f}%, "
          f"LS={acc_spread_list[-1]*100:.1f}%, "
          f"Supervised={acc_supervised[-1]*100:.1f}%")

# Full supervised baseline
lr_full = LogisticRegression(max_iter=1000, random_state=42)
lr_full.fit(X_d, y_d)
acc_full = accuracy_score(y_d, lr_full.predict(X_d))
print(f"\n  100% labeled (full supervised): {acc_full*100:.1f}%")

# Best Label Propagation result
print("\n--- Final Evaluation (10% labeled data) ---")
n_lab_final  = int(0.10 * len(y_d))
lab_idx_final = np.random.choice(len(y_d), n_lab_final, replace=False)
y_partial_final = np.full(len(y_d), -1)
y_partial_final[lab_idx_final] = y_d[lab_idx_final]

lp_final = LabelPropagation(kernel='knn', n_neighbors=7, max_iter=1000)
lp_final.fit(X_d, y_partial_final)
y_pred_final = lp_final.predict(X_d)

unlabeled_mask = y_partial_final == -1
acc_unlabeled  = accuracy_score(y_d[unlabeled_mask], y_pred_final[unlabeled_mask])
print(f"Accuracy on UNLABELED points: {acc_unlabeled*100:.1f}%")
print("\nClassification Report (on unlabeled points):")
print(classification_report(y_d[unlabeled_mask], y_pred_final[unlabeled_mask]))

# =============================================================================
# VISUALIZATION
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Label Propagation - Semi-Supervised Learning", fontsize=14, fontweight='bold')

# 1. Labeled points only (moons)
axes[0, 0].scatter(X_moons[:, 0], X_moons[:, 1], c='lightgray', s=15, alpha=0.5)
axes[0, 0].scatter(X_moons[labeled_idx, 0], X_moons[labeled_idx, 1],
                   c=y_moons[labeled_idx], cmap='bwr', s=100, edgecolors='black',
                   zorder=5, label='Labeled')
axes[0, 0].set_title(f"1. Only {n_labeled} Labels Available\n({n_labeled/n_total*100:.0f}% labeled)")
axes[0, 0].legend()

# 2. Supervised baseline (only trained on labeled)
axes[0, 1].scatter(X_moons[:, 0], X_moons[:, 1], c=y_pred_base,
                   cmap='bwr', alpha=0.5, s=15)
axes[0, 1].scatter(X_moons[labeled_idx, 0], X_moons[labeled_idx, 1],
                   c=y_moons[labeled_idx], cmap='bwr', s=100, edgecolors='black', zorder=5)
axes[0, 1].set_title(f"2. Supervised Only\nAcc={acc_base*100:.1f}% (trained on {n_labeled} pts)")

# 3. Label Propagation result
axes[0, 2].scatter(X_moons[:, 0], X_moons[:, 1], c=y_pred_moons,
                   cmap='bwr', alpha=0.5, s=15)
axes[0, 2].scatter(X_moons[labeled_idx, 0], X_moons[labeled_idx, 1],
                   c=y_moons[labeled_idx], cmap='bwr', s=100, edgecolors='black', zorder=5)
axes[0, 2].set_title(f"3. Label Propagation\nAcc={acc_lp*100:.1f}% (uses all {n_total} pts)")

# 4. Accuracy vs label fraction
label_pcts = [f * 100 for f in label_fractions]
axes[1, 0].plot(label_pcts, [a*100 for a in acc_lp_list],      'b-o', lw=2, label='Label Propagation')
axes[1, 0].plot(label_pcts, [a*100 for a in acc_spread_list],  'g-s', lw=2, label='Label Spreading')
axes[1, 0].plot(label_pcts, [a*100 for a in acc_supervised],   'r-^', lw=2, label='Supervised Only')
axes[1, 0].axhline(acc_full*100, color='black', linestyle='--', label=f'Full Supervised ({acc_full*100:.1f}%)')
axes[1, 0].set_title("4. Accuracy vs. % Labeled Data")
axes[1, 0].set_xlabel("% of Data Labeled")
axes[1, 0].set_ylabel("Accuracy (%)")
axes[1, 0].legend(); axes[1, 0].grid(True)

# 5. Label Propagation probability distribution
prob_sample = lp_final.label_distributions_[unlabeled_mask][:5]
axes[1, 1].imshow(prob_sample, cmap='Blues', aspect='auto', vmin=0, vmax=1)
axes[1, 1].set_title("5. Label Probabilities\n(5 unlabeled samples, 10 classes)")
axes[1, 1].set_xlabel("Class (Digit)")
axes[1, 1].set_ylabel("Sample")
axes[1, 1].set_xticks(range(10))
plt.colorbar(plt.cm.ScalarMappable(cmap='Blues'), ax=axes[1, 1], label='Probability')

# 6. Summary text
axes[1, 2].axis('off')
summary = (
    "KEY INSIGHTS\n"
    "─────────────────────────────────────\n\n"
    "Semi-supervised learning leverages\n"
    "unlabeled data to improve accuracy\n"
    "when labeled data is scarce.\n\n"
    f"Moons Dataset (only {n_labeled} labels):\n"
    f"  Supervised  : {acc_base*100:.1f}%\n"
    f"  Label Prop  : {acc_lp*100:.1f}%\n\n"
    f"Digits Dataset (10% labeled):\n"
    f"  Supervised  : {acc_supervised[2]*100:.1f}%\n"
    f"  Label Prop  : {acc_lp_list[2]*100:.1f}%\n"
    f"  Full Labels : {acc_full*100:.1f}%\n\n"
    "Algorithm: Labels 'flow' through the\n"
    "graph from labeled to similar\n"
    "unlabeled points."
)
axes[1, 2].text(0.05, 0.95, summary, transform=axes[1, 2].transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig("label_propagation_results.png", dpi=150)
plt.show()
print("\nPlot saved as 'label_propagation_results.png'")
