import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# ============================================
# SETTINGS
# ============================================

DATASET_PATH = "ai_detection_dataset"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 123

# ============================================
# LOAD DATASET
# ============================================

print("===================================")
print("LOADING AI DETECTION DATASET")
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
print("Class Names:", class_names)

# ============================================
# DATA AUGMENTATION
# ============================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.05),
    layers.RandomContrast(0.05)
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

print("Class weights:", class_weights)

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

# Initially freeze pretrained layers
base_model.trainable = False

# ============================================
# BUILD MODEL
# ============================================

inputs = layers.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

# MobileNetV2 preprocessing
x = tf.keras.applications.mobilenet_v2.preprocess_input(
    x * 255.0
)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.5)(x)

outputs = layers.Dense(
    2,
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
# MODEL SUMMARY
# ============================================

print()
print("===================================")
print("MODEL SUMMARY")
print("===================================")

model.summary()

# ============================================
# CREATE MODEL FOLDER
# ============================================

os.makedirs(
    "model",
    exist_ok=True
)

best_model_path = (
    "model/ai_detector_best.keras"
)

# ============================================
# CALLBACKS
# ============================================

early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True,
    verbose=1
)

checkpoint = callbacks.ModelCheckpoint(
    best_model_path,
    monitor="val_loss",
    save_best_only=True,
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
# TRAIN CLASSIFIER HEAD
# ============================================

print()
print("===================================")
print("TRAINING MOBILENETV2 CLASSIFIER")
print("===================================")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[
        early_stopping,
        checkpoint,
        reduce_lr
    ]
)

# ============================================
# LOAD BEST MODEL
# ============================================

print()
print("Loading best model...")

model = tf.keras.models.load_model(
    best_model_path
)

# ============================================
# EVALUATE
# ============================================

print()
print("===================================")
print("FINAL EVALUATION")
print("===================================")

validation_loss, validation_accuracy = model.evaluate(
    validation_dataset
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

final_model_path = (
    "model/ai_detector.keras"
)

model.save(final_model_path)

print()
print("===================================")
print("MODEL SAVED SUCCESSFULLY!")
print("===================================")
print(
    f"Location : "
    f"{final_model_path}"
)
print("===================================")

# ============================================
# ACCURACY GRAPH
# ============================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "AI Image Detection - Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig(
    "model/ai_detection_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================
# LOSS GRAPH
# ============================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "AI Image Detection - Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig(
    "model/ai_detection_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print()
print("Graphs saved successfully!")