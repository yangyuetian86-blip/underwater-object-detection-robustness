import cv2
import numpy as np
from pathlib import Path
import random
import shutil


def apply_blur(image, level="medium"):
    """
    Random-direction motion blur.
    """

    if level == "low":
        kernel_size = 15
    elif level == "medium":
        kernel_size = 31
    else:
        kernel_size = 45

    angle = random.uniform(0, 180)

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1

    center = (kernel_size / 2, kernel_size / 2)

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1
    )

    kernel = cv2.warpAffine(
        kernel,
        rotation_matrix,
        (kernel_size, kernel_size)
    )

    kernel_sum = np.sum(kernel)
    if kernel_sum > 0:
        kernel /= kernel_sum

    blurred = cv2.filter2D(
        image,
        -1,
        kernel
    )

    return blurred


def add_haze(image, level="medium"):
    """
    Deterministic haze simulation.
    """

    haze_map = {
        "low": 0.20,
        "medium": 0.40,
        "high": 0.60
    }

    intensity = haze_map[level]

    haze_layer = np.full(
        image.shape,
        255,
        dtype=np.uint8
    )

    hazy = cv2.addWeighted(
        image,
        1 - intensity,
        haze_layer,
        intensity,
        0
    )

    return hazy


def generate_robustness_dataset():
    PROJECT_ROOT = Path(r"F:/Program Files/Underwater_Detection")

    DATASET_ROOT = PROJECT_ROOT / "yolo_dataset"

    INPUT_IMAGE_DIR = DATASET_ROOT / "images/test"
    INPUT_LABEL_DIR = DATASET_ROOT / "labels/test"

    OUTPUT_ROOT = DATASET_ROOT / "benchmark"

    levels = ["low", "medium", "high"]
    methods = ["blur", "haze", "blur_haze"]

    # Create folders
    for method in methods:
        for level in levels:
            out_img_dir = OUTPUT_ROOT / f"{method}_{level}" / "images"
            out_label_dir = OUTPUT_ROOT / f"{method}_{level}" / "labels"

            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_label_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(INPUT_IMAGE_DIR.glob("*.jpg"))

    print(f"Found test images: {len(image_paths)}")

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Warning: failed to read {image_path}")
            continue

        filename = image_path.name
        stem = image_path.stem

        label_path = INPUT_LABEL_DIR / f"{stem}.txt"

        if not label_path.exists():
            print(f"Warning: missing label {label_path}")
            continue

        for level in levels:
            # Blur
            blur_img = apply_blur(image, level)

            blur_img_path = OUTPUT_ROOT / f"blur_{level}" / "images" / filename
            blur_label_path = OUTPUT_ROOT / f"blur_{level}" / "labels" / f"{stem}.txt"

            cv2.imwrite(str(blur_img_path), blur_img)
            shutil.copy2(label_path, blur_label_path)

            # Haze
            haze_img = add_haze(image, level)

            haze_img_path = OUTPUT_ROOT / f"haze_{level}" / "images" / filename
            haze_label_path = OUTPUT_ROOT / f"haze_{level}" / "labels" / f"{stem}.txt"

            cv2.imwrite(str(haze_img_path), haze_img)
            shutil.copy2(label_path, haze_label_path)

            # Blur + Haze
            blur_haze_img = apply_blur(image, level)
            blur_haze_img = add_haze(blur_haze_img, level)

            blur_haze_img_path = OUTPUT_ROOT / f"blur_haze_{level}" / "images" / filename
            blur_haze_label_path = OUTPUT_ROOT / f"blur_haze_{level}" / "labels" / f"{stem}.txt"

            cv2.imwrite(str(blur_haze_img_path), blur_haze_img)
            shutil.copy2(label_path, blur_haze_label_path)

    print("Robustness benchmark generated.")
    print(f"Output root: {OUTPUT_ROOT}")


if __name__ == "__main__":
    generate_robustness_dataset()