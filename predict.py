import tensorflow as tf
import numpy as np
import os
import sys

# ============================================
# SETTINGS
# ============================================

WILDLIFE_MODEL_PATH = "model/wildlife_classifier.keras"
AI_MODEL_PATH = "model/ai_detector.keras"

IMAGE_SIZE = (224, 224)

# These are the classes used by your wildlife dataset
CLASS_NAMES = [
    "butterfly",
    "cat",
    "cow",
    "dog",
    "elephant",
    "hen",
    "horse",
    "sheep",
    "spider",
    "squirrel"
]

# AI detector class order
AI_CLASSES = [
    "AI Generated",
    "Real"
]

# ============================================
# CHECK IMAGE ARGUMENT
# ============================================

if len(sys.argv) < 2:
    print()
    print("===================================")
    print("WILDLIFE IMAGE CLASSIFIER")
    print("===================================")
    print()
    print("Usage:")
    print("python predict.py image.jpg")
    print()
    print("Example:")
    print("python predict.py test.jpg")
    print()
    sys.exit()

image_path = sys.argv[1]

if not os.path.exists(image_path):
    print()
    print("ERROR: Image not found!")
    print()
    print("Image path:")
    print(image_path)
    print()
    sys.exit()

# ============================================
# CHECK MODELS
# ============================================

if not os.path.exists(WILDLIFE_MODEL_PATH):
    print()
    print("ERROR: Wildlife model not found!")
    print(WILDLIFE_MODEL_PATH)
    print()
    sys.exit()

if not os.path.exists(AI_MODEL_PATH):
    print()
    print("ERROR: AI detector model not found!")
    print(AI_MODEL_PATH)
    print()
    sys.exit()

# ============================================
# LOAD MODELS
# ============================================

print()
print("Loading models...")

wildlife_model = tf.keras.models.load_model(
    WILDLIFE_MODEL_PATH
)

ai_model = tf.keras.models.load_model(
    AI_MODEL_PATH
)

print("Models loaded successfully!")

# ============================================
# LOAD IMAGE
# ============================================

image = tf.keras.utils.load_img(
    image_path,
    target_size=IMAGE_SIZE
)

image_array = tf.keras.utils.img_to_array(
    image
)

# Keep a copy for the wildlife classifier
wildlife_image = image_array.copy()

# ============================================
# WILDLIFE CLASSIFICATION
# ============================================

# Your wildlife model was trained with images
# normalized using 1/255.
wildlife_image = wildlife_image / 255.0

wildlife_image = np.expand_dims(
    wildlife_image,
    axis=0
)

wildlife_prediction = wildlife_model.predict(
    wildlife_image,
    verbose=0
)

wildlife_probabilities = wildlife_prediction[0]

animal_index = np.argmax(
    wildlife_probabilities
)

animal_name = CLASS_NAMES[animal_index]

animal_confidence = (
    wildlife_probabilities[animal_index] * 100
)

# ============================================
# AI IMAGE DETECTION
# ============================================

# The MobileNetV2 AI detector contains its own
# preprocessing layer, so use the original
# 0-255 image here.

ai_image = np.expand_dims(
    image_array,
    axis=0
)

ai_prediction = ai_model.predict(
    ai_image,
    verbose=0
)

ai_probabilities = ai_prediction[0]

ai_probability = ai_probabilities[0]
real_probability = ai_probabilities[1]

if ai_probability > real_probability:

    image_type = "AI GENERATED"
    ai_confidence = ai_probability * 100

else:

    image_type = "REAL"
    ai_confidence = real_probability * 100

# ============================================
# DISPLAY FINAL RESULT
# ============================================

print()
print("==========================================")
print("       WILDLIFE IMAGE ANALYSIS")
print("==========================================")

print()
print("Input Image")
print("------------------------------------------")
print(image_path)

print()
print("WILDLIFE CLASSIFICATION")
print("------------------------------------------")
print(f"Animal             : {animal_name}")
print(f"Confidence         : {animal_confidence:.2f}%")

print()
print("AI IMAGE DETECTION")
print("------------------------------------------")
print(f"AI Generated       : {ai_probability * 100:.2f}%")
print(f"Real               : {real_probability * 100:.2f}%")

print()
print(f"Image Type         : {image_type}")
print(f"Confidence         : {ai_confidence:.2f}%")

print()
print("==========================================")
print("              FINAL RESULT")
print("==========================================")

print(
    f"Animal             : "
    f"{animal_name.upper()}"
)

print(
    f"Image Type         : "
    f"{image_type}"
)

print(
    f"Animal Confidence  : "
    f"{animal_confidence:.2f}%"
)

print(
    f"AI Confidence      : "
    f"{ai_confidence:.2f}%"
)

print("==========================================")