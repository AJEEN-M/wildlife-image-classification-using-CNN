import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

    # ============================================
    # PAGE SETTINGS
    # ============================================

st.set_page_config(
    page_title="Wildlife Image Analyzer",
    page_icon="🐾",
    layout="centered"
)

    # ============================================
    # SETTINGS
    # ============================================

IMAGE_SIZE = (224, 224)

WILDLIFE_MODEL_PATH = "model/wildlife_classifier_v2.keras"
AI_MODEL_PATH = "model/ai_detector.keras"

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

    # ============================================
    # LOAD MODELS
    # ============================================

@st.cache_resource
def load_models():

    print("Loading wildlife model...")

    wildlife_model = tf.keras.models.load_model(
        WILDLIFE_MODEL_PATH
    )

    print("Wildlife model loaded successfully!")

    print("Loading AI detection model...")

    ai_model = tf.keras.models.load_model(
        AI_MODEL_PATH
    )

    print("AI detection model loaded successfully!")

    return wildlife_model, ai_model


wildlife_model, ai_model = load_models()

# ============================================
# TITLE
# ============================================

st.title("🐾 Wildlife Image Analyzer")

st.write(
    "Upload a wildlife image to identify the animal "
    "and determine whether the image is real or AI-generated."
)

st.divider()

# ============================================
# IMAGE UPLOAD
# ============================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# ============================================
# PROCESS IMAGE
# ============================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.divider()

    if st.button(
        "🔍 Analyze Image",
        use_container_width=True
    ):

        # ========================================
        # PREPARE IMAGE
        # ========================================

        resized_image = image.resize(
            IMAGE_SIZE
        )

        image_array = np.array(
            resized_image
        ).astype("float32")

        # ========================================
        # WILDLIFE CLASSIFICATION
        # ========================================

        # IMPORTANT:
        #
        # The V2 wildlife model contains
        # MobileNetV2 preprocess_input internally.
        #
        # Therefore, DO NOT divide by 255 here.
        #
        # The model expects the image in the
        # original 0-255 range.

        wildlife_input = np.expand_dims(
            image_array,
            axis=0
        )

        wildlife_prediction = (
            wildlife_model.predict(
                wildlife_input,
                verbose=0
            )
        )

        wildlife_probabilities = (
            wildlife_prediction[0]
        )

        # ----------------------------------------
        # FIND PREDICTED ANIMAL
        # ----------------------------------------

        animal_index = np.argmax(
            wildlife_probabilities
        )

        animal_name = CLASS_NAMES[
            animal_index
        ]

        animal_confidence = float(
            wildlife_probabilities[
                animal_index
            ] * 100
        )

        # ========================================
        # AI IMAGE DETECTION
        # ========================================

        # Keep the original AI detector
        # preprocessing unchanged.

        ai_input = np.expand_dims(
            image_array,
            axis=0
        )

        ai_prediction = ai_model.predict(
            ai_input,
            verbose=0
        )

        ai_probabilities = (
            ai_prediction[0]
        )

        ai_probability = float(
            ai_probabilities[0]
        )

        real_probability = float(
            ai_probabilities[1]
        )

        # ----------------------------------------
        # DETERMINE IMAGE TYPE
        # ----------------------------------------

        if ai_probability > real_probability:

            image_type = "AI GENERATED"

            image_confidence = (
                ai_probability * 100
            )

        else:

            image_type = "REAL"

            image_confidence = (
                real_probability * 100
            )

        # ========================================
        # DISPLAY RESULTS
        # ========================================

        st.subheader(
            "Prediction Results"
        )

        # ========================================
        # ANIMAL CLASSIFICATION
        # ========================================

        st.markdown(
            "### 🐾 Animal Classification"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Animal",
                animal_name.upper()
            )

        with col2:

            st.metric(
                "Confidence",
                f"{animal_confidence:.2f}%"
            )

        st.progress(
            min(
                animal_confidence / 100,
                1.0
            )
        )

        # ========================================
        # IMAGE AUTHENTICITY
        # ========================================

        st.markdown(
            "### 🤖 Image Authenticity"
        )

        col3, col4 = st.columns(2)

        with col3:

            if image_type == "AI GENERATED":

                st.error(
                    "AI GENERATED"
                )

            else:

                st.success(
                    "REAL IMAGE"
                )

        with col4:

            st.metric(
                "Confidence",
                f"{image_confidence:.2f}%"
            )

        st.progress(
            min(
                image_confidence / 100,
                1.0
            )
        )

        # ========================================
        # DETAILED PROBABILITIES
        # ========================================

        with st.expander(
            "View detailed probabilities"
        ):

            st.write(
                f"**Animal:** "
                f"{animal_name.upper()}"
            )

            st.write(
                f"**Animal confidence:** "
                f"{animal_confidence:.2f}%"
            )

            st.write(
                f"**AI Generated:** "
                f"{ai_probability * 100:.2f}%"
            )

            st.write(
                f"**Real:** "
                f"{real_probability * 100:.2f}%"
            )

        # ========================================
        # FOOTER
        # ========================================

        st.divider()

        st.caption(
            "This system uses deep learning models "
            "trained on wildlife and real/AI-generated "
            "image datasets."
        )
#new feature added