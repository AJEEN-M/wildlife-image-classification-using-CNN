import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# ============================================
# SETTINGS
# ============================================

DATASET_PATH = "dataset_balanced"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 123

MODEL_PATH = "model/wildlife_classifier.keras"

# ============================================
# LOAD DATASET
# ============================================

print()
print("===================================")
print("LOADING WILDLIFE DATASET")
print("===================================")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names

print()
print("Classes:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

NUM_CLASSES = len(class_names)

# ============================================
# CLASS WEIGHTS
# ============================================

print()
print("Calculating class weights...")

labels = []

for images, batch_labels in train_dataset:
    labels.extend(batch_labels.numpy())

labels = np.array(labels)

classes = np.unique(labels)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=labels
)

class_weights = {
    int(classes[i]): float(weights[i])
    for i in range(len(classes))
}

print("Class weights:")
print(class_weights)

# ============================================
# DATA AUGMENTATION
# ============================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10)
])

# ============================================
# PREFETCH
# ============================================

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    tf.data.AUTOTUNE
)

# ============================================
# LOAD MOBILENETV2
# ============================================

print()
print("===================================")
print("LOADING MOBILENETV2")
print("===================================")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze base model initially
base_model.trainable = False

# ============================================
# BUILD MODEL
# ============================================

inputs = layers.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(
    x * 255.0
)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(
    256,
    activation="relu"
)(x)

x = layers.Dropout(0.4)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs
)

# ============================================
# COMPILE
# ============================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ============================================
# SUMMARY
# ============================================

print()
print("===================================")
print("MODEL SUMMARY")
print("===================================")

model.summary()

# ============================================
# MODEL FOLDER
# ============================================

os.makedirs(
    "model",
    exist_ok=True
)

BEST_MODEL_PATH = (
    "model/wildlife_classifier_best.keras"
)

# ============================================
# CALLBACKS
# ============================================

checkpoint = callbacks.ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stopping = callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.000001,
    verbose=1
)

# ============================================
# STAGE 1 TRAINING
# ============================================

print()
print("===================================")
print("STAGE 1: TRAINING CLASSIFIER")
print("===================================")

history1 = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)

# ============================================
# FINE-TUNING
# ============================================

print()
print("===================================")
print("STAGE 2: FINE-TUNING MOBILENETV2")
print("===================================")

base_model.trainable = True

# Freeze most layers.
# Only the final 30 layers will be fine-tuned.
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    initial_epoch=len(history1.history["loss"]),
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)

# ============================================
# LOAD BEST MODEL
# ============================================

print()
print("===================================")
print("LOADING BEST MODEL")
print("===================================")

model = tf.keras.models.load_model(
    BEST_MODEL_PATH
)

# ============================================
# FINAL EVALUATION
# ============================================

print()
print("===================================")
print("FINAL EVALUATION")
print("===================================")

validation_loss, validation_accuracy = (
    model.evaluate(
        validation_dataset
    )
)

print()
print(
    f"Validation Loss     : "
    f"{validation_loss:.4f}"
)

print(
    f"Validation Accuracy : "
    f"{validation_accuracy * 100:.2f}%"
)

# ============================================
# SAVE FINAL MODEL
# ============================================

model.save(
    MODEL_PATH
)

print()
print("===================================")
print("MODEL SAVED SUCCESSFULLY!")
print("===================================")

print(
    f"Location : {MODEL_PATH}"
)

print("===================================")

# ============================================
# COMBINE TRAINING HISTORY
# ============================================

accuracy = (
    history1.history["accuracy"]
    + history2.history["accuracy"]
)

val_accuracy = (
    history1.history["val_accuracy"]
    + history2.history["val_accuracy"]
)

loss = (
    history1.history["loss"]
    + history2.history["loss"]
)

val_loss = (
    history1.history["val_loss"]
    + history2.history["val_loss"]
)

# ============================================
# ACCURACY GRAPH
# ============================================

plt.figure(figsize=(8, 5))

plt.plot(
    accuracy,
    label="Training Accuracy"
)

plt.plot(
    val_accuracy,
    label="Validation Accuracy"
)

plt.title(
    "Wildlife Classification - Accuracy"
)

plt.xlabel("Epoch") 
plt.ylabel("Accuracy")

plt.legend()

plt.savefig(
    "model/wildlife_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================
# LOSS GRAPH
# ============================================

plt.figure(figsize=(8, 5))

plt.plot(
    loss,
    label="Training Loss"
)

plt.plot(
    val_loss,
    label="Validation Loss"
)

plt.title(
    "Wildlife Classification - Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.savefig(
    "model/wildlife_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print()
print("===================================")
print("GRAPHS SAVED")
print("===================================")
print("Accuracy : model/wildlife_accuracy.png")
print("Loss     : model/wildlife_loss.png")