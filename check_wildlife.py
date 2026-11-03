import tensorflow as tf
import numpy as np

DATASET_PATH = "dataset"
MODEL_PATH = "model/wildlife_classifier.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123

print()
print("======================================")
print("LOADING VALIDATION DATASET")
print("======================================")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

class_names = validation_dataset.class_names

print()
print("Classes:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

print()
print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# ============================================
# PREDICTIONS
# ============================================

print()
print("Generating predictions...")

all_true = []
all_predicted = []

for images, labels in validation_dataset:

    images = images / 255.0

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    all_true.extend(labels.numpy())
    all_predicted.extend(predicted_classes)

all_true = np.array(all_true)
all_predicted = np.array(all_predicted)

# ============================================
# CONFUSION MATRIX
# ============================================

cm = tf.math.confusion_matrix(
    all_true,
    all_predicted,
    num_classes=len(class_names)
).numpy()

print()
print("======================================")
print("CONFUSION MATRIX")
print("======================================")

print(
    "Actual \\ Predicted"
)

print(
    " " * 18 +
    " ".join(
        f"{i:>6}"
        for i in range(len(class_names))
    )
)

for i, row in enumerate(cm):

    print(
        f"{i:>2} {class_names[i]:<14}" +
        " ".join(
            f"{value:>6}"
            for value in row
        )
    )

# ============================================
# CLASS-WISE ACCURACY
# ============================================

print()
print("======================================")
print("CLASS-WISE ACCURACY")
print("======================================")

for i, class_name in enumerate(class_names):

    total = np.sum(cm[i])
    correct = cm[i][i]

    accuracy = (
        correct / total * 100
        if total > 0
        else 0
    )

    print(
        f"{class_name:<12} : "
        f"{accuracy:.2f}% "
        f"({correct}/{total})"
    )

# ============================================
# MOST COMMON CONFUSIONS
# ============================================

print()
print("======================================")
print("MOST COMMON CONFUSIONS")
print("======================================")

confusions = []

for actual in range(len(class_names)):

    for predicted in range(len(class_names)):

        if actual != predicted:

            count = cm[actual][predicted]

            if count > 0:
                confusions.append(
                    (
                        count,
                        class_names[actual],
                        class_names[predicted]
                    )
                )

confusions.sort(reverse=True)

for count, actual, predicted in confusions[:20]:

    print(
        f"{actual:<12} → "
        f"{predicted:<12} : "
        f"{count}"
    )

print()
print("======================================")
print("DIAGNOSTIC COMPLETE")
print("======================================")