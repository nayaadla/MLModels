# =============================================================================
# LOGISTIC REGRESSION
# =============================================================================
# Logistic Regression is a supervised learning algorithm for CLASSIFICATION.
# Despite its name, it predicts the PROBABILITY of a class using the sigmoid
# function: σ(z) = 1 / (1 + e^(-z)), where z = wX + b
#
# Unlike linear regression (continuous output), logistic regression outputs
# a probability between 0 and 1, then applies a threshold (usually 0.5).
#
# Key Concepts:
#   - Sigmoid Function: maps any value to (0, 1)
#   - Cost Function: Binary Cross-Entropy (Log Loss)
#   - Decision Boundary: the line/plane separating classes
#   - Evaluation: Accuracy, Precision, Recall, F1-Score, ROC-AUC
#   - Multiclass: One-vs-Rest (OvR) strategy
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# -----------------------------------------------------------------------------
# 1. LOAD DATASET
# -----------------------------------------------------------------------------
# Breast Cancer Wisconsin Dataset:
#   - 30 features: radius, texture, perimeter, area, smoothness, etc.
#   - Binary target: Malignant (1) or Benign (0)
#   - 569 samples
print("=" * 60)
print("LOGISTIC REGRESSION - Breast Cancer Dataset")
print("=" * 60)

data = load_breast_cancer()
X, y = data.data, data.target

print(f"\nDataset shape  : {X.shape}")
print(f"Classes        : {data.target_names}  (0=Malignant, 1=Benign)")
print(f"Class counts   : Malignant={sum(y==0)}, Benign={sum(y==1)}")

# -----------------------------------------------------------------------------
# 2. PREPROCESS
# -----------------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")

# -----------------------------------------------------------------------------
# 3. TRAIN MODEL
# -----------------------------------------------------------------------------
# C = inverse of regularization strength (higher C = less regularization)
# max_iter increased for convergence
model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# 4. EVALUATE
# -----------------------------------------------------------------------------
y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]   # Probability of class 1

accuracy = accuracy_score(y_test, y_pred)
roc_auc  = roc_auc_score(y_test, y_pred_prob)

print("\n--- Evaluation Metrics ---")
print(f"  Accuracy : {accuracy*100:.2f}%")
print(f"  ROC-AUC  : {roc_auc:.4f}")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# -----------------------------------------------------------------------------
# 5. VISUALIZE
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Logistic Regression - Breast Cancer", fontsize=14, fontweight='bold')

# Plot 1: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=data.target_names, yticklabels=data.target_names)
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")
axes[0].set_title("Confusion Matrix")

# Plot 2: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
axes[1].plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC Curve (AUC = {roc_auc:.3f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random Classifier')
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate (Recall)")
axes[1].set_title("ROC Curve")
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.savefig("logistic_regression_results.png", dpi=150)
plt.show()
print("\nPlot saved as 'logistic_regression_results.png'")
