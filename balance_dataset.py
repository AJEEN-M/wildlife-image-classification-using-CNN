import os
import random
import shutil

SOURCE = "dataset"
DESTINATION = "dataset_balanced"

CLASSES = [
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

TARGET_COUNT = 1400

random.seed(42)

os.makedirs(DESTINATION, exist_ok=True)

print()
print("======================================")
print("CREATING BALANCED WILDLIFE DATASET")
print("======================================")

for class_name in CLASSES:

    source_folder = os.path.join(
        SOURCE,
        class_name
    )

    destination_folder = os.path.join(
        DESTINATION,
        class_name
    )

    os.makedirs(
        destination_folder,
        exist_ok=True
    )

    images = []

    for filename in os.listdir(source_folder):

        filepath = os.path.join(
            source_folder,
            filename
        )

        if os.path.isfile(filepath):

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension in [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp"
            ]:
                images.append(filename)

    random.shuffle(images)

    selected = images[:TARGET_COUNT]

    print(
        f"{class_name:10s} : "
        f"{len(images)} available → "
        f"{len(selected)} selected"
    )

    for filename in selected:

        source_file = os.path.join(
            source_folder,
            filename
        )

        destination_file = os.path.join(
            destination_folder,
            filename
        )

        shutil.copy2(
            source_file,
            destination_file
        )

print()
print("======================================")
print("BALANCED DATASET CREATED")
print("======================================")

print(
    f"Location : {DESTINATION}"
)

print(
    f"Images   : {TARGET_COUNT * len(CLASSES)}"
)

print("======================================")