# =============================================================================
# SELF-TRAINING - Semi-Supervised Learning
# =============================================================================
# Self-Training (also called Self-Learning or Pseudo-Labeling) is the simplest
# semi-supervised learning approach. It uses a base classifier iteratively:
#
# Algorithm:
#   1. Train a base classifier on the LABELED data only
#   2. Predict labels (with confidence scores) for UNLABELED data
#   3. Select the most confident predictions (above a threshold)
#   4. Add these pseudo-labeled samples to the training set
#   5. Retrain the classifier on original labels + pseudo labels
#   6. Repeat until no more samples are added or max iterations reached
#
# Key Design Choices:
#   - Base Classifier: any probabilistic classifier (SVM, RF, LogReg, etc.)
#   - Confidence Threshold: how sure must the model be? (e.g., 0.9 = 90%)
#     * Too low → adds incorrect pseudo-labels → error propagation
#     * Too high → adds very few samples → slow progress
#   - Selection Strategy: fixed threshold OR top-k per class per iteration
#
# Why it works:
#   - High-confidence predictions are likely correct
#   - More training data helps the model generalize better
#   - Unlabeled data reveals the underlying data distribution
#
# Limitations:
#   - Error propagation: early mistakes can compound
#   - Biased toward majority classes
#   - Depends heavily on quality of initial labeled set
#
# Comparison with Label Propagation:
#   - Label Propagation: graph-based, propagates labels through similarity graph
#   - Self-Training: wrapper method, works with ANY base classifier
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

np.random.seed(42)

print("=" * 60)
print("SELF-TRAINING - Semi-Supervised Learning")
print("=" * 60)

# =============================================================================
# PART A: Step-by-step illustration of Self-Training
# =============================================================================
print("\n--- Part A: Self-Training on Synthetic Data ---")

X_syn, y_syn = make_classification(
    n_samples=500, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, random_state=42
)
X_syn = StandardScaler().fit_transform(X_syn)

# Simulate 5% labeled data
n_labeled = 25  # only 25 out of 500
labeled_idx   = np.random.choice(len(y_syn), n_labeled, replace=False)
unlabeled_idx = np.setdiff1d(np.arange(len(y_syn)), labeled_idx)

y_partial_syn             = np.full(len(y_syn), -1)
y_partial_syn[labeled_idx] = y_syn[labeled_idx]

print(f"Total samples    : {len(y_syn)}")
print(f"Labeled samples  : {n_labeled} ({n_labeled/len(y_syn)*100:.0f}%)")
print(f"Unlabeled samples: {len(unlabeled_idx)} ({len(unlabeled_idx)/len(y_syn)*100:.0f}%)")

# Manual self-training loop to track progress
def manual_self_training(X, y_partial, threshold=0.9, max_iter=20):
    """Run self-training manually to track which samples get pseudo-labeled."""
    y_current  = y_partial.copy()
    labeled    = y_partial != -1
    history    = {'n_labeled': [labeled.sum()], 'accuracy': []}
    iteration  = 0

    base = LogisticRegression(max_iter=500, random_state=42)

    while iteration < max_iter:
        X_lab = X[labeled]
        y_lab = y_current[labeled]
        base.fit(X_lab, y_lab)

        unlabeled_mask  = y_current == -1
        if unlabeled_mask.sum() == 0:
            break

        probs      = base.predict_proba(X[unlabeled_mask])
        max_probs  = probs.max(axis=1)
        pseudo_labels = base.predict(X[unlabeled_mask])

        # Add samples above threshold
        confident  = max_probs >= threshold
        if not confident.any():
            print(f"  Iter {iteration+1}: No confident predictions. Stopping.")
            break

        unlabeled_indices = np.where(unlabeled_mask)[0]
        new_labeled_idx   = unlabeled_indices[confident]
        y_current[new_labeled_idx] = pseudo_labels[confident]
        labeled[new_labeled_idx]   = True

        acc = accuracy_score(y_syn, base.predict(X))
        history['n_labeled'].append(labeled.sum())
        history['accuracy'].append(acc)

        print(f"  Iter {iteration+1:2d}: Added {confident.sum():3d} pseudo-labels "
              f"| Total labeled: {labeled.sum():3d} | Acc: {acc*100:.1f}%")
        iteration += 1

    return base, history, y_current

print("\nManual Self-Training (threshold=0.9):")
model_manual, history, y_final = manual_self_training(X_syn, y_partial_syn)

# =============================================================================
# PART B: sklearn SelfTrainingClassifier on Digits
# =============================================================================
print("\n--- Part B: Digits Dataset with SelfTrainingClassifier ---")

digits  = load_digits()
X_d     = StandardScaler().fit_transform(digits.data)
y_d     = digits.target

label_fractions  = [0.05, 0.10, 0.20, 0.30, 0.50]
results = {
    'SVM-SelfTrain':    [],
    'RF-SelfTrain':     [],
    'SVM-Supervised':   [],
    'RF-Supervised':    [],
}

for frac in label_fractions:
    # Create partial labels
    n_lab   = max(int(frac * len(y_d)), 20)
    lab_idx = []
    for cls in np.unique(y_d):
        cls_idx  = np.where(y_d == cls)[0]
        n_sel    = max(1, int(frac * len(cls_idx)))
        lab_idx.extend(np.random.choice(cls_idx, n_sel, replace=False))
    lab_idx   = np.array(lab_idx)
    y_partial = np.full(len(y_d), -1)
    y_partial[lab_idx] = y_d[lab_idx]

    # SVM Self-Training
    svm_base = SVC(probability=True, kernel='rbf', C=10, gamma='scale', random_state=42)
    svm_st   = SelfTrainingClassifier(svm_base, threshold=0.85, max_iter=10, verbose=False)
    svm_st.fit(X_d, y_partial)
    results['SVM-SelfTrain'].append(accuracy_score(y_d, svm_st.predict(X_d)))

    # RF Self-Training
    rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_st   = SelfTrainingClassifier(rf_base, threshold=0.85, max_iter=10, verbose=False)
    rf_st.fit(X_d, y_partial)
    results['RF-SelfTrain'].append(accuracy_score(y_d, rf_st.predict(X_d)))

    # Supervised baselines (labeled only)
    svm_sup = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm_sup.fit(X_d[lab_idx], y_d[lab_idx])
    results['SVM-Supervised'].append(accuracy_score(y_d, svm_sup.predict(X_d)))

    rf_sup = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_sup.fit(X_d[lab_idx], y_d[lab_idx])
    results['RF-Supervised'].append(accuracy_score(y_d, rf_sup.predict(X_d)))

    print(f"  {frac*100:.0f}% labeled ({len(lab_idx)} pts): "
          f"SVM-ST={results['SVM-SelfTrain'][-1]*100:.1f}% | "
          f"RF-ST={results['RF-SelfTrain'][-1]*100:.1f}% | "
          f"SVM-Sup={results['SVM-Supervised'][-1]*100:.1f}%")

# Full supervised
svm_full = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm_full.fit(X_d, y_d)
acc_full = accuracy_score(y_d, svm_full.predict(X_d))
print(f"\n  Full supervised (100%): {acc_full*100:.1f}%")

# =============================================================================
# VISUALIZATION
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Self-Training - Semi-Supervised Learning", fontsize=14, fontweight='bold')

# 1. Initial labeled points
axes[0, 0].scatter(X_syn[:, 0], X_syn[:, 1], c='lightgray', s=15, alpha=0.4)
axes[0, 0].scatter(X_syn[labeled_idx, 0], X_syn[labeled_idx, 1],
                   c=y_syn[labeled_idx], cmap='bwr', s=100,
                   edgecolors='black', zorder=5)
axes[0, 0].set_title(f"1. Start: Only {n_labeled} Labeled Points\n(gray = unlabeled)")

# 2. After self-training
labeled_after  = y_final != -1
pseudo_mask    = labeled_after & (y_partial_syn == -1)
axes[0, 1].scatter(X_syn[~labeled_after, 0], X_syn[~labeled_after, 1],
                   c='lightgray', s=15, alpha=0.4)
axes[0, 1].scatter(X_syn[pseudo_mask, 0], X_syn[pseudo_mask, 1],
                   c=y_final[pseudo_mask], cmap='bwr', alpha=0.4, s=20,
                   label=f'Pseudo-labeled ({pseudo_mask.sum()})')
axes[0, 1].scatter(X_syn[labeled_idx, 0], X_syn[labeled_idx, 1],
                   c=y_syn[labeled_idx], cmap='bwr', s=100,
                   edgecolors='black', zorder=5, label=f'Original ({n_labeled})')
axes[0, 1].set_title("2. After Self-Training\n(pseudo-labels added)")
axes[0, 1].legend(fontsize=8)

# 3. Growth of labeled set
axes[0, 2].plot(history['n_labeled'], 'b-o', lw=2, markersize=6)
axes[0, 2].axhline(len(y_syn), color='green', linestyle='--', label='All samples')
axes[0, 2].set_title("3. Growth of Labeled Set per Iteration")
axes[0, 2].set_xlabel("Iteration"); axes[0, 2].set_ylabel("# Labeled Samples")
axes[0, 2].legend(); axes[0, 2].grid(True)

# 4. Accuracy vs fraction labeled
label_pcts = [f * 100 for f in label_fractions]
axes[1, 0].plot(label_pcts, [a*100 for a in results['SVM-SelfTrain']],  'b-o', lw=2, label='SVM + Self-Training')
axes[1, 0].plot(label_pcts, [a*100 for a in results['RF-SelfTrain']],   'g-s', lw=2, label='RF + Self-Training')
axes[1, 0].plot(label_pcts, [a*100 for a in results['SVM-Supervised']], 'b--^', lw=2, alpha=0.6, label='SVM Supervised Only')
axes[1, 0].plot(label_pcts, [a*100 for a in results['RF-Supervised']],  'g--^', lw=2, alpha=0.6, label='RF Supervised Only')
axes[1, 0].axhline(acc_full*100, color='red', linestyle=':', label=f'Full Supervised ({acc_full*100:.1f}%)')
axes[1, 0].set_title("4. Accuracy vs. % Labeled Data")
axes[1, 0].set_xlabel("% of Data Labeled"); axes[1, 0].set_ylabel("Accuracy (%)")
axes[1, 0].legend(fontsize=8); axes[1, 0].grid(True)

# 5. Accuracy over iterations (manual)
if history['accuracy']:
    axes[1, 1].plot(range(1, len(history['accuracy'])+1), [a*100 for a in history['accuracy']],
                    'purple', lw=2, marker='o')
    axes[1, 1].set_title("5. Accuracy Growth per Iteration\n(Synthetic Data, threshold=0.9)")
    axes[1, 1].set_xlabel("Iteration"); axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].grid(True)

# 6. Algorithm summary
axes[1, 2].axis('off')
summary = (
    "SELF-TRAINING ALGORITHM\n"
    "─────────────────────────────────────\n\n"
    "1. Train classifier on labeled data\n\n"
    "2. Predict unlabeled data\n"
    "   → get confidence scores\n\n"
    "3. Add high-confidence predictions\n"
    "   as 'pseudo-labels'\n\n"
    "4. Retrain with enlarged dataset\n\n"
    "5. Repeat until convergence\n\n"
    "─────────────────────────────────────\n"
    "KEY PARAMETER: threshold\n"
    "  Too low  → noisy pseudo-labels\n"
    "  Too high → few new samples\n\n"
    "vs. Label Propagation:\n"
    "  + Works with ANY classifier\n"
    "  - Error propagation risk"
)
axes[1, 2].text(0.05, 0.95, summary, transform=axes[1, 2].transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig("self_training_results.png", dpi=150)
plt.show()
print("\nPlot saved as 'self_training_results.png'")
