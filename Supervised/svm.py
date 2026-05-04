# =============================================================================
# SUPPORT VECTOR MACHINE (SVM)
# =============================================================================
# SVM is a supervised learning algorithm that finds the OPTIMAL HYPERPLANE
# that best separates classes by maximizing the MARGIN between them.
#
# Key Concepts:
#   - Support Vectors: the data points closest to the decision boundary
#   - Margin: the distance between the hyperplane and the nearest support vectors
#     SVM maximizes this margin → better generalization
#   - Kernel Trick: maps data to higher dimensions to find a linear separator
#     in cases where data is not linearly separable in original space
#
# Kernels:
#   - Linear    : K(x,z) = x·z              (linearly separable data)
#   - RBF/Gaussian: K(x,z) = exp(-γ||x-z||²) (non-linear, most common)
#   - Polynomial: K(x,z) = (x·z + c)^d
#
# Hyperparameters:
#   - C    : Regularization (high C = less margin, fits training data more)
#   - gamma: Influence of each training example (high = more complex boundary)
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

# -----------------------------------------------------------------------------
# 1. LOAD DATASET
# -----------------------------------------------------------------------------
# Digits Dataset:
#   - 8x8 pixel images of handwritten digits (0-9)
#   - 1797 samples, 64 features (flattened pixels)
#   - 10 classes
print("=" * 60)
print("SUPPORT VECTOR MACHINE (SVM) - Digits Dataset")
print("=" * 60)

data = load_digits()
X, y = data.data, data.target

print(f"\nDataset shape  : {X.shape}")
print(f"Number of classes: {len(np.unique(y))} (digits 0-9)")
print(f"Samples per class: ~{len(y)//10}")

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
# RBF kernel works well for this image classification task
# C=10, gamma='scale' are good starting hyperparameters
print("\nTraining SVM with RBF kernel...")
model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True)
model.fit(X_train, y_train)

print(f"Number of support vectors: {model.n_support_.sum()}")
print(f"Support vectors per class: {model.n_support_}")

# -----------------------------------------------------------------------------
# 4. EVALUATE
# -----------------------------------------------------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Evaluation Metrics ---")
print(f"  Accuracy: {accuracy*100:.2f}%")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------------------------------------------------------
# 5. VISUALIZE
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("SVM - Handwritten Digits Classification", fontsize=14, fontweight='bold')

# Plot 1: Sample images
sample_axes = plt.subplot2grid((1, 3), (0, 0))
fig2, sample_axes_grid = plt.subplots(2, 5, figsize=(10, 4))
fig2.suptitle("Sample Digit Images", fontsize=12)
for digit in range(10):
    idx = np.where(y == digit)[0][0]
    row, col = divmod(digit, 5)
    sample_axes_grid[row, col].imshow(data.images[idx], cmap='gray')
    sample_axes_grid[row, col].set_title(f"Digit: {digit}")
    sample_axes_grid[row, col].axis('off')
plt.tight_layout()
plt.savefig("svm_digit_samples.png", dpi=150)
plt.close(fig2)

# Plot 2: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")
axes[0].set_title("Confusion Matrix")

# Plot 3: Decision boundary visualization using PCA (2D)
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y, test_size=0.2, random_state=42, stratify=y
)
model_2d = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
model_2d.fit(X_train_2d, y_train_2d)

x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

axes[1].contourf(xx, yy, Z, alpha=0.3, cmap='tab10')
scatter = axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', s=10, alpha=0.6)
axes[1].set_title("Decision Regions (PCA 2D projection)")
axes[1].set_xlabel("PCA Component 1")
axes[1].set_ylabel("PCA Component 2")

# Plot 4: Misclassified examples
wrong_idx = np.where(y_pred != y_test)[0]
axes[2].axis('off')
axes[2].set_title(f"Misclassified: {len(wrong_idx)}/{len(y_test)} samples\nAccuracy: {accuracy*100:.2f}%",
                  fontsize=12)

plt.tight_layout()
plt.savefig("svm_results.png", dpi=150)
plt.show()
print("\nPlots saved as 'svm_results.png' and 'svm_digit_samples.png'")
