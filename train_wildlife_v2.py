import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ============================================
# SETTINGS
# ============================================

DATASET_PATH = "dataset"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123

EPOCHS_HEAD = 8
EPOCHS_FINE = 12

OUTPUT_MODEL = "model/wildlife_classifier_v2.keras"
BEST_MODEL = "model/wildlife_classifier_v2_best.keras"

# ============================================
# LOAD DATASET
# ============================================

print()
print("======================================")
print("LOADING WILDLIFE DATASET")
print("======================================")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_dataset.class_names
num_classes = len(class_names)

print()
print("Classes:")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")

print()
print(f"Number of classes: {num_classes}")

# ============================================
# DATA AUGMENTATION
# ============================================

augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.10, 0.10),
    layers.RandomContrast(0.10),
], name="data_augmentation")

# ============================================
# PERFORMANCE
# ============================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    AUTOTUNE
)

# ============================================
# MOBILENETV2
# ============================================

print()
print("======================================")
print("LOADING MOBILENETV2")
print("======================================")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# ============================================
# BUILD MODEL
# ============================================

inputs = layers.Input(
    shape=(224, 224, 3)
)

x = augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(
    x
)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dense(
    256,
    activation="relu"
)(x)

x = layers.Dropout(0.4)(x)

outputs = layers.Dense(
    num_classes,
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
        learning_rate=0.0003
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ============================================
# CALLBACKS
# ============================================

os.makedirs(
    "model",
    exist_ok=True
)

checkpoint = callbacks.ModelCheckpoint(
    BEST_MODEL,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

early_stopping = callbacks.EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=4,
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
# STAGE 1
# ============================================

print()
print("======================================")
print("STAGE 1: TRAINING CLASSIFICATION HEAD")
print("======================================")

history_head = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS_HEAD,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)

# ============================================
# STAGE 2: FINE-TUNING
# ============================================

print()
print("======================================")
print("STAGE 2: FINE-TUNING")
print("======================================")

base_model.trainable = True

# Freeze most MobileNetV2 layers.
# Fine-tune only the final 40 layers.
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS_FINE,
    initial_epoch=len(
        history_head.history["loss"]
    ),
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
print("======================================")
print("LOADING BEST MODEL")
print("======================================")

model = tf.keras.models.load_model(
    BEST_MODEL
)

# ============================================
# FINAL EVALUATION
# ============================================

print()
print("======================================")
print("FINAL EVALUATION")
print("======================================")

loss, accuracy = model.evaluate(
    validation_dataset,
    verbose=1
)

print()
print(
    f"Validation Loss     : {loss:.4f}"
)

print(
    f"Validation Accuracy : {accuracy * 100:.2f}%"
)

# ============================================
# SAVE MODEL
# ============================================

model.save(
    OUTPUT_MODEL
)

print()
print("======================================")
print("MODEL SAVED")
print("======================================")

print(
    f"Location : {OUTPUT_MODEL}"
)

print()
print("IMPORTANT:")
print(
    "The original wildlife_classifier.keras "
    "has NOT been changed."
)

print("======================================")