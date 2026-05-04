# =============================================================================
# SIMPLE CONVOLUTIONAL NEURAL NETWORK (CNN)
# =============================================================================
# CNNs are a class of deep learning models designed for processing structured
# grid data like images. They use convolutional layers to automatically learn
# spatial features (edges, shapes, textures) from raw pixels.
#
# Architecture Layers:
#   1. Conv2D     : Applies learnable filters to detect local patterns
#                  Output size = (W - F + 2P) / S + 1
#                  W=input size, F=filter size, P=padding, S=stride
#   2. BatchNorm  : Normalizes activations → stable, faster training
#   3. ReLU       : Activation function: max(0, x) → adds non-linearity
#   4. MaxPooling : Downsamples feature maps → reduces spatial dimensions
#   5. Dropout    : Randomly zeros neurons during training → prevents overfitting
#   6. Flatten    : Converts 2D feature maps to 1D vector
#   7. Dense      : Fully connected layer for final classification
#
# Key Concepts:
#   - Feature Maps: output of each conv layer
#   - Receptive Field: region of input that influences a neuron
#   - Parameter Sharing: same filter weights applied across entire image
#   - Pooling: spatial invariance, dimensionality reduction
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------------------------------------------------------
# 1. LOAD & PREPROCESS DATASET
# -----------------------------------------------------------------------------
# CIFAR-10: 60,000 color images (32x32x3) in 10 classes
# Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
print("=" * 60)
print("SIMPLE CNN - CIFAR-10 Image Classification")
print("=" * 60)

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(f"\nTraining set shape: {X_train.shape}")
print(f"Test set shape    : {X_test.shape}")
print(f"Image shape       : {X_train.shape[1:]} (H x W x Channels)")
print(f"Pixel value range : {X_train.min()} - {X_train.max()}")

# Normalize pixel values from [0, 255] to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32') / 255.0

# One-hot encode labels: e.g., class 3 → [0,0,0,1,0,0,0,0,0,0]
y_train_cat = to_categorical(y_train, 10)
y_test_cat  = to_categorical(y_test, 10)

print(f"\nAfter normalization: {X_train.min():.1f} - {X_train.max():.1f}")
print(f"Label shape (one-hot): {y_train_cat.shape}")

# -----------------------------------------------------------------------------
# 2. VISUALIZE SAMPLE IMAGES
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("CIFAR-10 Sample Images", fontsize=13, fontweight='bold')
for i, ax in enumerate(axes.flat):
    idx = np.where(y_train.flatten() == i)[0][0]
    ax.imshow(X_train[idx])
    ax.set_title(CLASS_NAMES[i])
    ax.axis('off')
plt.tight_layout()
plt.savefig("cnn_samples.png", dpi=150)
plt.show()

# -----------------------------------------------------------------------------
# 3. BUILD CNN MODEL
# -----------------------------------------------------------------------------
def build_cnn():
    model = keras.Sequential([
        # --- Block 1: Detect low-level features (edges, colors) ---
        layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                      input_shape=(32, 32, 3)),
        # 32 filters, 3x3 kernel, 'same' padding preserves spatial size
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),        # 32x32 → 16x16
        layers.Dropout(0.25),

        # --- Block 2: Detect mid-level features (shapes, textures) ---
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),        # 16x16 → 8x8
        layers.Dropout(0.25),

        # --- Block 3: Detect high-level features (object parts) ---
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),        # 8x8 → 4x4
        layers.Dropout(0.25),

        # --- Classifier Head ---
        layers.Flatten(),                   # 4x4x128 = 2048 neurons
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')  # 10 classes, probabilities sum to 1
    ])
    return model

model = build_cnn()
model.summary()

# -----------------------------------------------------------------------------
# 4. COMPILE & TRAIN
# -----------------------------------------------------------------------------
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',   # multi-class log loss
    metrics=['accuracy']
)

# Callbacks: stop early if no improvement, reduce LR on plateau
callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True,
                                   monitor='val_accuracy'),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
]

print("\nTraining CNN...")
history = model.fit(
    X_train, y_train_cat,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------------------------------------------------------
# 5. EVALUATE
# -----------------------------------------------------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")
print(f"Test Loss    : {test_loss:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=1)
print("\nClassification Report:")
print(classification_report(y_test.flatten(), y_pred, target_names=CLASS_NAMES))

# -----------------------------------------------------------------------------
# 6. VISUALIZE RESULTS
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"Simple CNN - CIFAR-10 (Test Acc: {test_acc*100:.2f}%)",
             fontsize=14, fontweight='bold')

# Training curves
axes[0].plot(history.history['accuracy'],     label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[0].set_title("Accuracy over Epochs")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
axes[0].legend(); axes[0].grid(True)

axes[1].plot(history.history['loss'],     label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Val Loss')
axes[1].set_title("Loss over Epochs")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].legend(); axes[1].grid(True)

# Confusion matrix
cm = confusion_matrix(y_test.flatten(), y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[2].set_title("Confusion Matrix")
axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("True")
plt.setp(axes[2].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig("cnn_results.png", dpi=150)
plt.show()

model.save("simple_cnn_cifar10.h5")
print("\nModel saved as 'simple_cnn_cifar10.h5'")
print("Plot saved as 'cnn_results.png'")
