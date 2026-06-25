# =================================================================== #
# File: generate_augmented_dataset.py                                 #
# Description: Script to generate augmented dataset for underwater    #
# object detection, including CLAHE enhancement, Multi Scale Retinex, #
# Color Jitter, Blur Augmentation, and Haze Augmentation. The script  #
# applies a random augmentation strategy to each image in the original#
# dataset and saves the augmented images and corresponding labels     #
# in a new directory.                                                 #
# Author: Yuntian Xiao                                                #
# =================================================================== #

import cv2
import numpy as np
import random
from pathlib import Path
import shutil

# from kfold_train import PROJECT_ROOT

# CLAHE Enhancement
def apply_clahe(image):
    """
    Apply CLAHE enhancement on luminance channel.
    """

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    cl = clahe.apply(l)

    augmented_lab = cv2.merge((cl, a, b))

    augmented_image = cv2.cvtColor(
        augmented_lab,
        cv2.COLOR_LAB2BGR
    )

    return augmented_image

# Multi Scale Retinex
def apply_msr(
    image,
    sigmas=[30, 80, 150]
):

    image = image.astype(np.float32) + 1.0

    msr = np.zeros_like(image)

    # Multi-scale Retinex
    for sigma in sigmas:
        blur = cv2.GaussianBlur(
            image,
            (0, 0),
            sigma
        )

        retinex = np.log(image) - np.log(blur)

        msr += retinex

    # Average scales
    msr = msr / len(sigmas)

    # Global normalization
    msr = cv2.normalize(
        msr,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    msr = np.clip(
        msr,
        0,
        255
    ).astype(np.uint8)

    # Gamma correction
    gamma = 1.6

    msr = msr.astype(np.float32) / 255.0
    msr = np.power(msr, gamma)
    msr = np.clip(
        msr * 255,
        0,
        255
    ).astype(np.uint8)

    return msr

# Color Jitter
def apply_color_jitter(image):
    """
    Color jitter implementation.
    """

    image = image.astype(np.float32) / 255.0

    # Brightness
    brightness_factor = random.uniform(0.9, 1.1)
    image = image * brightness_factor

    # Contrast
    contrast_factor = random.uniform(0.9, 1.1)
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    image = (image - mean) * contrast_factor + mean

    # Clip before HSV conversion
    image = np.clip(image, 0, 1)

    # Convert to uint8 safely
    image_uint8 = (image * 255).astype(np.uint8)

    # HSV saturation
    hsv = cv2.cvtColor(
        image_uint8,
        cv2.COLOR_BGR2HSV
    )

    saturation_factor = random.uniform(0.9, 1.1)

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1].astype(np.float32)
        * saturation_factor,
        0,
        255
    ).astype(np.uint8)

    image = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )

    return image

# Blur Augmentation
def apply_blur(image):
    """
    Apply Gaussian blur or motion blur.
    """

    choice = random.choice(["gaussian", "motion"])

    if choice == "gaussian":

        blurred = cv2.GaussianBlur(
            image,
            (5, 5),
            0
        )

    else:

        kernel_size = 9

        kernel = np.zeros(
            (kernel_size, kernel_size)
        )

        kernel[
            int((kernel_size - 1) / 2),
            :
        ] = np.ones(kernel_size)

        kernel = kernel / kernel_size

        blurred = cv2.filter2D(
            image,
            -1,
            kernel
        )

    return blurred

# Haze Augmentation
def add_haze(image, intensity=0.2):
    """
    Simulate underwater haze effect.
    """

    h, w, _ = image.shape

    haze_layer = np.full(
        (h, w, 3),
        255,
        dtype=np.uint8
    )

    hazy_image = cv2.addWeighted(
        image,
        1 - intensity,
        haze_layer,
        intensity,
        0
    )

    return hazy_image


# Pipeline
def underwater_pipeline(image):
    """
    Apply a random augmentation strategy to the input image. Strategies include:
    - Original: No augmentation, return None to indicate skipping.
    - Restoration: Apply either CLAHE or Multi Scale Retinex.
    - Degradation: Apply blur and/or haze.
    - Mixed: Apply a combination of CLAHE, color jitter, blur, and haze.
    """
    
    # Random strategy selection
    strategy = random.choice([
        "original",
        "restoration",
        "degradation",
        "mixed"
    ])

    # Original
    if strategy == "original":
        return None

    # Restoration
    elif strategy == "restoration":
        if random.random() < 0.2:
            output = apply_msr(image)
        else:
            output = apply_clahe(image)

    # Degradation
    elif strategy == "degradation":
        output = image

        if random.random() < 0.7:
            output = apply_blur(output.copy() if output is image else output)

        if random.random() < 0.7:
            haze_intensity = random.uniform(0.1, 0.3)
            target = output.copy() if output is image else output
            output = add_haze(target, haze_intensity)

    # Mixed
    else:
        output = apply_clahe(image)

        if random.random() < 0.5:
            output = apply_color_jitter(output)

        if random.random() < 0.5:
            output = apply_blur(output)

        if random.random() < 0.5:
            haze_intensity = random.uniform(0.1, 0.2)
            output = add_haze(output, haze_intensity)

    return output

# augmentation dataset generation
def generate_dataset():

    # Dataset paths
    PROJECT_ROOT = Path(
        r"F:/Program Files/Underwater_Detection"
    )

    ORIGINAL_DATASET_ROOT = PROJECT_ROOT / "yolo_dataset"
    AUG_DATASET_ROOT = PROJECT_ROOT / "yolo_dataset_augmented"

    # 原始 train 数据
    INPUT_IMAGE_DIR = ORIGINAL_DATASET_ROOT / "images/train"
    INPUT_LABEL_DIR = ORIGINAL_DATASET_ROOT / "labels/train"

    # 原始 test 数据：不增强，只复制
    TEST_IMAGE_DIR = ORIGINAL_DATASET_ROOT / "images/test"
    TEST_LABEL_DIR = ORIGINAL_DATASET_ROOT / "labels/test"

    # 增强版数据集输出路径
    OUTPUT_IMAGE_DIR = AUG_DATASET_ROOT / "images/train"
    OUTPUT_LABEL_DIR = AUG_DATASET_ROOT / "labels/train"

    OUTPUT_TEST_IMAGE_DIR = AUG_DATASET_ROOT / "images/test"
    OUTPUT_TEST_LABEL_DIR = AUG_DATASET_ROOT / "labels/test"

    # 如果之前生成过增强数据集，先删除，避免旧文件混入
    if AUG_DATASET_ROOT.exists():
        print(f"删除旧增强数据集: {AUG_DATASET_ROOT}")
        shutil.rmtree(AUG_DATASET_ROOT)

    # Create output folders
    for p in [
        OUTPUT_IMAGE_DIR,
        OUTPUT_LABEL_DIR,
        OUTPUT_TEST_IMAGE_DIR,
        OUTPUT_TEST_LABEL_DIR,
    ]:
        p.mkdir(
            parents=True,
            exist_ok=True
        )

    # ============================================================
    # 1. 复制原始 train 图像和标签
    # ============================================================
    print("正在复制原始 train 数据...")

    image_paths = sorted(
        INPUT_IMAGE_DIR.glob("*.jpg")
    )

    copied_train_count = 0

    for image_path in image_paths:
        stem = image_path.stem
        label_path = INPUT_LABEL_DIR / f"{stem}.txt"

        if not label_path.exists():
            print(f"Missing train label: {label_path}")
            continue

        shutil.copy2(
            image_path,
            OUTPUT_IMAGE_DIR / image_path.name
        )

        shutil.copy2(
            label_path,
            OUTPUT_LABEL_DIR / label_path.name
        )

        copied_train_count += 1

    print(f"原始 train 数据复制完成: {copied_train_count} 张")

    # ============================================================
    # 2. 复制固定 test 图像和标签
    # ============================================================
    print("正在复制固定 test 数据...")

    test_image_paths = sorted(
        TEST_IMAGE_DIR.glob("*.jpg")
    )

    copied_test_count = 0

    for image_path in test_image_paths:
        stem = image_path.stem
        label_path = TEST_LABEL_DIR / f"{stem}.txt"

        if not label_path.exists():
            print(f"Missing test label: {label_path}")
            continue

        shutil.copy2(
            image_path,
            OUTPUT_TEST_IMAGE_DIR / image_path.name
        )

        shutil.copy2(
            label_path,
            OUTPUT_TEST_LABEL_DIR / label_path.name
        )

        copied_test_count += 1

    print(f"固定 test 数据复制完成: {copied_test_count} 张")

    # ============================================================
    # 3. 对 train 图像生成增强版本
    # ============================================================
    print("正在生成 train 增强图像...")

    augmented_count = 0
    skipped_count = 0

    for idx, image_path in enumerate(image_paths):
        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(f"Cannot load: {image_path}")
            continue

        stem = image_path.stem

        input_label_path = INPUT_LABEL_DIR / f"{stem}.txt"

        if not input_label_path.exists():
            print(f"Missing label: {input_label_path}")
            continue

        # Apply augmentation pipeline
        augmented = underwater_pipeline(image)

        # If strategy is "original", skip augmentation and do not save
        if augmented is None:
            skipped_count += 1
            continue

        # Save augmented image with new name
        output_image_path = (
            OUTPUT_IMAGE_DIR /
            f"{stem}_aug.jpg"
        )

        cv2.imwrite(
            str(output_image_path),
            augmented
        )

        # Copy corresponding label with new name
        output_label_path = (
            OUTPUT_LABEL_DIR /
            f"{stem}_aug.txt"
        )

        shutil.copy2(
            input_label_path,
            output_label_path
        )

        augmented_count += 1

        # Show progress
        if idx % 100 == 0:
            print(
                f"[{idx}/{len(image_paths)}] processed..."
            )

    print("Augmented dataset generation completed.")
    print(f"原始 train 数量: {copied_train_count}")
    print(f"增强 train 数量: {augmented_count}")
    print(f"跳过增强数量: {skipped_count}")
    print(f"固定 test 数量: {copied_test_count}")
    print(f"输出位置: {AUG_DATASET_ROOT}")


if __name__ == "__main__":
    generate_dataset()