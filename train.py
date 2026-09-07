"""
train.py
--------
Trains a MobileNetV2-based CNN for crop disease classification.

Steps:
  1. python generate_dataset.py   ← create synthetic images
  2. python train.py              ← train & save model
  3. python app.py                ← run the web app

Output: model/crop_disease_model.h5
        model/class_names.json
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                         ModelCheckpoint)

# ─── Hyper-parameters ─────────────────────────────────────────────────────────
IMG_SIZE    = 128
BATCH_SIZE  = 32
EPOCHS      = 25          # increase for better accuracy on real data
FINE_TUNE_AT= 100         # unfreeze MobileNetV2 layers above this index
TRAIN_DIR   = "dataset/train"
TEST_DIR    = "dataset/test"
MODEL_DIR   = "model"
MODEL_PATH  = os.path.join(MODEL_DIR, "crop_disease_model.h5")
CLASSES_PATH= os.path.join(MODEL_DIR, "class_names.json")
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)


# ─── 1. Data Generators ───────────────────────────────────────────────────────
def build_generators():
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.10,
        zoom_range=0.20,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.80, 1.20],
        fill_mode="nearest",
    )
    val_aug = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_aug.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )
    test_gen = val_aug.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    return train_gen, test_gen


# ─── 2. Model Architecture ────────────────────────────────────────────────────
def build_model(num_classes: int) -> tf.keras.Model:
    """
    Transfer-learning approach:
      • MobileNetV2 backbone (pre-trained on ImageNet, top removed)
      • Custom classifier head
      • Two-phase training: freeze backbone → fine-tune upper layers
    """
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False   # Phase 1: only train the head

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.45)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base


# ─── 3. Training ──────────────────────────────────────────────────────────────
def train():
    print("\n🌿 Crop Disease Detection — Model Training\n" + "=" * 45)

    # --- generators ---
    train_gen, test_gen = build_generators()
    num_classes = len(train_gen.class_indices)
    class_names = {v: k for k, v in train_gen.class_indices.items()}

    print(f"  Classes ({num_classes}): {list(train_gen.class_indices.keys())}")
    print(f"  Train samples : {train_gen.samples}")
    print(f"  Test  samples : {test_gen.samples}\n")

    # --- model ---
    model, base = build_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=6,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, min_lr=1e-6, verbose=1),
        ModelCheckpoint(MODEL_PATH, save_best_only=True,
                        monitor="val_accuracy", verbose=1),
    ]

    # ── Phase 1: train head only ──
    print("\n📌 Phase 1 — Training classification head …")
    history1 = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=min(EPOCHS, 15),
        callbacks=callbacks,
        verbose=1,
    )

    # ── Phase 2: fine-tune upper backbone layers ──
    print(f"\n📌 Phase 2 — Fine-tuning backbone from layer {FINE_TUNE_AT} …")
    base.trainable = True
    for layer in base.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),   # lower LR for fine-tuning
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history2 = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=EPOCHS,
        initial_epoch=len(history1.history["loss"]),
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ──
    print("\n📊 Final Evaluation on Test Set:")
    loss, acc = model.evaluate(test_gen, verbose=0)
    print(f"   Loss     : {loss:.4f}")
    print(f"   Accuracy : {acc * 100:.2f}%")

    # ── Save class names ──
    with open(CLASSES_PATH, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"\n✅ Model saved  → {MODEL_PATH}")
    print(f"✅ Classes saved → {CLASSES_PATH}")


if __name__ == "__main__":
    train()
