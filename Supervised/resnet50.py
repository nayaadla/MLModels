# =============================================================================
# RESNET50 - TRANSFER LEARNING
# =============================================================================
# ResNet50 (Residual Network with 50 layers) was introduced by He et al. (2015)
# and won the ImageNet challenge. It solves the "vanishing gradient" problem
# that prevented training very deep networks.
#
# Key Innovation — Residual/Skip Connections:
#   Instead of learning H(x) directly, layers learn the RESIDUAL F(x) = H(x) - x
#   Output: H(x) = F(x) + x
#   This allows gradients to flow directly through skip connections during
#   backpropagation, enabling training of very deep networks (50, 101, 152 layers)
#
# Architecture:
#   - Input: 224x224x3 image
#   - Conv1: 7x7, 64 filters, stride 2
#   - MaxPool: 3x3, stride 2
#   - 4 Residual Stages: [3, 4, 6, 3] bottleneck blocks
#   - GlobalAvgPool → Dense(1000) → Softmax (original ImageNet classes)
#
# Transfer Learning Strategy:
#   - Use pretrained ImageNet weights (feature extractor)
#   - Freeze base layers (don't update their weights)
#   - Replace final Dense layer with our custom classifier
#   - Fine-tune top layers for our specific task
#
# Why Transfer Learning?
#   - ResNet50 trained on 1.2M images for weeks — we reuse that knowledge
#   - Works well even with small datasets
#   - Much faster to train (only train the new head)
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------------------------------------------------------
# 1. LOAD & PREPROCESS DATASET
# -----------------------------------------------------------------------------
print("=" * 60)
print("RESNET50 TRANSFER LEARNING - CIFAR-10")
print("=" * 60)

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# ResNet50 expects at minimum 32x32 input, but performs better with 224x224
# We'll resize using tf.image for compatibility
# Normalize to [-1, 1] range (as used during ImageNet training of ResNet)
print(f"\nOriginal image shape: {X_train.shape[1:]}")
print("Resizing images to 64x64 for ResNet50 compatibility...")

def preprocess(X, y, img_size=64):
    """Resize images and apply ResNet50 preprocessing."""
    X = tf.image.resize(X, [img_size, img_size])
    X = keras.applications.resnet50.preprocess_input(X)  # scales to [-1, 1]
    return X.numpy(), y

X_train_r, y_train_r = preprocess(X_train, y_train)
X_test_r,  y_test_r  = preprocess(X_test,  y_test)

y_train_cat = to_categorical(y_train_r, 10)
y_test_cat  = to_categorical(y_test_r, 10)

print(f"Resized image shape : {X_train_r.shape[1:]}")

# -----------------------------------------------------------------------------
# 2. BUILD MODEL WITH TRANSFER LEARNING
# -----------------------------------------------------------------------------
def build_resnet50_model(img_size=64, num_classes=10):
    """
    Build ResNet50-based classifier using transfer learning.

    Phase 1: Feature Extraction
        - Load ResNet50 pretrained on ImageNet
        - Freeze all base layers
        - Add custom classification head

    Phase 2: Fine-Tuning (optional)
        - Unfreeze top layers of ResNet50
        - Train with very low learning rate
    """
    # Load ResNet50 WITHOUT the top classification layer (include_top=False)
    base_model = ResNet50(
        weights='imagenet',            # pretrained weights from ImageNet
        include_top=False,             # remove the 1000-class head
        input_shape=(img_size, img_size, 3)
    )

    # Freeze all base model layers (don't update ImageNet weights in phase 1)
    base_model.trainable = False

    print(f"\nResNet50 base model:")
    print(f"  Total layers       : {len(base_model.layers)}")
    print(f"  Trainable params   : {base_model.count_params():,}")

    # Build the full model
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)    # training=False keeps BN frozen
    x = layers.GlobalAveragePooling2D()(x)    # reduces to 1D (replaces Flatten)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model, base_model

IMG_SIZE = 64
model, base_model = build_resnet50_model(img_size=IMG_SIZE)
model.summary()

total_params    = model.count_params()
trainable       = sum([tf.size(w).numpy() for w in model.trainable_weights])
non_trainable   = total_params - trainable
print(f"\nTotal params      : {total_params:,}")
print(f"Trainable params  : {trainable:,}  (only our custom head)")
print(f"Non-trainable     : {non_trainable:,} (frozen ResNet50 weights)")

# -----------------------------------------------------------------------------
# 3. PHASE 1: TRAIN CUSTOM HEAD
# -----------------------------------------------------------------------------
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                                   monitor='val_accuracy'),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-7)
]

print("\nPhase 1: Training custom head (base frozen)...")
history1 = model.fit(
    X_train_r, y_train_cat,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------------------------------------------------------
# 4. PHASE 2: FINE-TUNE TOP LAYERS OF RESNET50
# -----------------------------------------------------------------------------
print("\nPhase 2: Fine-tuning top layers of ResNet50...")

# Unfreeze the last residual block (stage 4)
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

trainable2 = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"Trainable params after unfreeze: {trainable2:,}")

# Use a much smaller learning rate to avoid destroying pretrained features
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    X_train_r, y_train_cat,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# Combine histories
combined_acc     = history1.history['accuracy']     + history2.history['accuracy']
combined_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
combined_loss    = history1.history['loss']         + history2.history['loss']
combined_val_loss= history1.history['val_loss']     + history2.history['val_loss']

# -----------------------------------------------------------------------------
# 5. EVALUATE
# -----------------------------------------------------------------------------
test_loss, test_acc = model.evaluate(X_test_r, y_test_cat, verbose=0)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")
print(f"Test Loss    : {test_loss:.4f}")

y_pred = np.argmax(model.predict(X_test_r), axis=1)
print("\nClassification Report:")
print(classification_report(y_test_r.flatten(), y_pred, target_names=CLASS_NAMES))

# -----------------------------------------------------------------------------
# 6. VISUALIZE
# -----------------------------------------------------------------------------
phase1_end = len(history1.history['accuracy'])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"ResNet50 Transfer Learning - CIFAR-10 (Test Acc: {test_acc*100:.2f}%)",
             fontsize=13, fontweight='bold')

# Training curves with phase boundary
axes[0].plot(combined_acc,     label='Train Accuracy')
axes[0].plot(combined_val_acc, label='Val Accuracy')
axes[0].axvline(phase1_end, color='red', linestyle='--', label='Fine-tune starts')
axes[0].set_title("Accuracy (Phase 1 + Fine-tuning)")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
axes[0].legend(); axes[0].grid(True)

axes[1].plot(combined_loss,     label='Train Loss')
axes[1].plot(combined_val_loss, label='Val Loss')
axes[1].axvline(phase1_end, color='red', linestyle='--', label='Fine-tune starts')
axes[1].set_title("Loss (Phase 1 + Fine-tuning)")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].legend(); axes[1].grid(True)

cm = confusion_matrix(y_test_r.flatten(), y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[2].set_title("Confusion Matrix")
axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("True")
plt.setp(axes[2].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig("resnet50_results.png", dpi=150)
plt.show()

model.save("resnet50_cifar10.h5")
print("\nModel saved as 'resnet50_cifar10.h5'")
print("Plot saved as 'resnet50_results.png'")
