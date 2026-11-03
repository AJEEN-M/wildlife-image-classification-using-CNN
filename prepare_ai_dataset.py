import os
import shutil
import random

SOURCE_DIR = "dataset"
DEST_DIR = "ai_detection_dataset/real"

IMAGES_PER_CLASS = 50

os.makedirs(DEST_DIR, exist_ok=True)

classes = [
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

for class_name in classes:

    source_folder = os.path.join(SOURCE_DIR, class_name)

    if not os.path.exists(source_folder):
        print(f"Folder not found: {source_folder}")
        continue

    images = [
        file for file in os.listdir(source_folder)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    selected_images = images[:IMAGES_PER_CLASS]

    for i, image in enumerate(selected_images):

        source_path = os.path.join(source_folder, image)

        new_name = f"{class_name}_{i}.jpg"
        destination_path = os.path.join(DEST_DIR, new_name)

        shutil.copy2(source_path, destination_path)

    print(f"{class_name}: copied {len(selected_images)} images")

print("\nReal image dataset created successfully!")