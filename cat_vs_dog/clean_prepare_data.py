import os
from pathlib import Path
import shutil

images_folder = Path(__file__).parent/"dogs-vs-cats"/"train"

def copy_images(images_folder:Path):
    """
    This function copy all cat and dog image in respective folders
    """
    os.makedirs(images_folder/"cats", exist_ok=True)
    os.makedirs(images_folder/"dogs", exist_ok=True)

    for images in images_folder.rglob("*"):
        if not images.is_file():
            continue
        if images.is_file():
            shutil.move(images, images_folder / ("cats" if images.name.startswith("cat") else "dogs"))
    print("complete")

copy_images(images_folder=images_folder)
