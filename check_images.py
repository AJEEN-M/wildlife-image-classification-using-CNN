import os
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# ============================================
# SETTINGS
# ============================================

DATASET_PATH = "dataset"

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

SAMPLES_PER_CLASS = 10

# ============================================
# COLLECT IMAGES
# ============================================

random.seed(42)

samples = {}

for class_name in CLASS_NAMES:

    folder = os.path.join(
        DATASET_PATH,
        class_name
    )

    files = []

    for filename in os.listdir(folder):

        path = os.path.join(
            folder,
            filename
        )

        if os.path.isfile(path):

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension in [
                ".jpg",
                ".jpeg",
                ".png",
                ".webp"
            ]:
                files.append(path)

    random.shuffle(files)

    samples[class_name] = files[
        :SAMPLES_PER_CLASS
    ]

# ============================================
# CREATE CONTACT SHEET FOR EACH CLASS
# ============================================

os.makedirs(
    "dataset_samples",
    exist_ok=True
)

print()
print("======================================")
print("CREATING DATASET SAMPLE SHEETS")
print("======================================")

for class_name in CLASS_NAMES:

    images = samples[class_name]

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(15, 6)
    )

    fig.suptitle(
        f"Dataset Check: {class_name.upper()}",
        fontsize=18
    )

    for ax, image_path in zip(
        axes.ravel(),
        images
    ):

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            ax.imshow(image)
            ax.axis("off")

            filename = os.path.basename(
                image_path
            )

            ax.set_title(
                filename[:18],
                fontsize=7
            )

        except Exception as e:

            ax.axis("off")
            print(
                f"Could not open: "
                f"{image_path}"
            )

    plt.tight_layout()

    output_path = os.path.join(
        "dataset_samples",
        f"{class_name}.jpg"
    )

    plt.savefig(
        output_path,
        dpi=120,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Created: {output_path}"
    )

print()
print("======================================")
print("DONE")
print("======================================")

print()
print(
    "Open the dataset_samples folder "
    "and inspect the 10 images for each class."
)