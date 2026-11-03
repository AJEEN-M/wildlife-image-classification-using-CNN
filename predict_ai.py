import tensorflow as tf
import numpy as np
import sys
import os

# ============================================
# SETTINGS
# ============================================

MODEL_PATH = "model/ai_detector.keras"
IMAGE_SIZE = (224, 224)

# ============================================
# CHECK IMAGE ARGUMENT
# ============================================

if len(sys.argv) < 2:
    print()
    print("Usage:")
    print("python predict_ai.py path_to_image")
    print()
    print("Example:")
    print("python predict_ai.py test.jpg")
    print()
    sys.exit()

image_path = sys.argv[1]

if not os.path.exists(image_path):
    print()
    print("ERROR: Image not found!")
    print(f"Path: {image_path}")
    print()
    sys.exit()

# ============================================
# LOAD MODEL
# ============================================

print()
print("Loading AI detector...")

model = tf.keras.models.load_model(MODEL_PATH)

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

image_array = np.expand_dims(
    image_array,
    axis=0
)

# ============================================
# PREDICTION
# ============================================

prediction = model.predict(
    image_array,
    verbose=0
)

# Dataset class order is alphabetical:
# ['ai_generated', 'real']

ai_probability = prediction[0][0]
real_probability = prediction[0][1]

# ============================================
# RESULT
# ============================================

if ai_probability > real_probability:

    result = "AI GENERATED"
    confidence = ai_probability * 100

else:

    result = "REAL"
    confidence = real_probability * 100

# ============================================
# DISPLAY RESULT
# ============================================

print()
print("===================================")
print("       AI IMAGE DETECTION")
print("===================================")

print(f"Image              : {image_path}")

print()
print(f"AI Generated       : {ai_probability * 100:.2f}%")
print(f"Real               : {real_probability * 100:.2f}%")

print()
print(f"Prediction         : {result}")
print(f"Confidence         : {confidence:.2f}%")

print("===================================")