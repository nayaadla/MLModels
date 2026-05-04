# =============================================================================
# LINEAR REGRESSION
# =============================================================================
# Linear Regression is a supervised learning algorithm that models the
# relationship between a dependent variable (y) and one or more independent
# variables (X) by fitting a linear equation: y = wX + b
#
# Goal: Predict a CONTINUOUS output value (e.g., house price, temperature)
#
# Key Concepts:
#   - Cost Function: Mean Squared Error (MSE) = (1/n) * sum((y_pred - y_true)^2)
#   - Optimization: Minimize MSE using gradient descent or closed-form solution
#   - Evaluation Metrics: MSE, RMSE, MAE, R² Score
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# 1. LOAD DATASET
# -----------------------------------------------------------------------------
# California Housing Dataset:
#   - Features: median income, house age, avg rooms, avg bedrooms, population, etc.
#   - Target: Median house value (in $100,000s)
print("=" * 60)
print("LINEAR REGRESSION - California Housing Dataset")
print("=" * 60)

data = fetch_california_housing()
X, y = data.data, data.target

print(f"\nDataset shape     : {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Feature names     : {list(data.feature_names)}")
print(f"Target range      : ${y.min()*100:.0f}k - ${y.max()*100:.0f}k")

# -----------------------------------------------------------------------------
# 2. PREPROCESS
# -----------------------------------------------------------------------------
# Standardize features so that gradient descent converges faster
# and coefficients are comparable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples  : {X_train.shape[0]}")
print(f"Testing samples   : {X_test.shape[0]}")

# -----------------------------------------------------------------------------
# 3. TRAIN MODEL
# -----------------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print(f"\nModel Intercept (b): {model.intercept_:.4f}")
print("\nFeature Coefficients (weights):")
for name, coef in zip(data.feature_names, model.coef_):
    print(f"  {name:<20}: {coef:+.4f}")

# -----------------------------------------------------------------------------
# 4. EVALUATE
# -----------------------------------------------------------------------------
y_pred = model.predict(X_test)

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("\n--- Evaluation Metrics ---")
print(f"  MSE  (Mean Squared Error)      : {mse:.4f}")
print(f"  RMSE (Root Mean Squared Error) : {rmse:.4f}  (~${rmse*100:.0f}k average error)")
print(f"  MAE  (Mean Absolute Error)     : {mae:.4f}")
print(f"  R²   (Coefficient of Det.)     : {r2:.4f}  ({r2*100:.1f}% variance explained)")

# -----------------------------------------------------------------------------
# 5. VISUALIZE
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Linear Regression - California Housing", fontsize=14, fontweight='bold')

# Plot 1: Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.3, color='steelblue', s=10)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel("Actual Price ($100k)")
axes[0].set_ylabel("Predicted Price ($100k)")
axes[0].set_title(f"Actual vs Predicted (R² = {r2:.3f})")
axes[0].legend()

# Plot 2: Residuals
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.3, color='coral', s=10)
axes[1].axhline(0, color='black', lw=2, linestyle='--')
axes[1].set_xlabel("Predicted Values")
axes[1].set_ylabel("Residuals (Actual - Predicted)")
axes[1].set_title("Residual Plot (should be random around 0)")

plt.tight_layout()
plt.savefig("linear_regression_results.png", dpi=150)
plt.show()
print("\nPlot saved as 'linear_regression_results.png'")
