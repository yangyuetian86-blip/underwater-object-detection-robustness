# =================================================================== #
# File: generate_robustness_benchmark.py                              #
# Description: Script to generate robustness benchmark for underwater #
# object detection, including Blur and Haze augmentation. The script  #
# applies a random augmentation strategy to each image in the original#
# dataset and saves the augmented images and corresponding labels     #
# in a new directory.                                                 #
# Author: Yuntian Xiao                                                #
# =================================================================== #

import cv2
import numpy as np
from pathlib import Path
import random

def apply_blur(
    image,
    level="medium"
):
    """
    Random-direction motion blur.
    """

    if level == "low":
        kernel_size = 15

    elif level == "medium":
        kernel_size = 31

    else:
        kernel_size = 45

    angle = random.uniform(
        0,
        180
    )

    kernel = np.zeros(
        (kernel_size, kernel_size),
        dtype=np.float32
    )

    kernel[
        kernel_size // 2,
        :
    ] = 1

    center = (
        kernel_size / 2,
        kernel_size / 2
    )

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

    kernel /= np.sum(kernel)

    blurred = cv2.filter2D(
        image,
        -1,
        kernel
    )

    return blurred

def add_haze(
    image,
    level="medium"
):
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

    PROJECT_ROOT = Path(
        r"F:/Program Files/Underwater_Detection"
    )

    DATASET_ROOT = (
        PROJECT_ROOT /
        "yolo_dataset"
    )

    INPUT_DIR = (
        DATASET_ROOT /
        "images/test"
    )

    OUTPUT_ROOT = (
        DATASET_ROOT /
        "benchmark"
    )

    levels = [
        "low",
        "medium",
        "high"
    ]

    for level in levels:

        (OUTPUT_ROOT /
         f"blur_{level}").mkdir(
            parents=True,
            exist_ok=True
        )

        (OUTPUT_ROOT /
         f"haze_{level}").mkdir(
            parents=True,
            exist_ok=True
        )

        (OUTPUT_ROOT /
         f"blur_haze_{level}").mkdir(
            parents=True,
            exist_ok=True
        )

    image_paths = sorted(
        INPUT_DIR.glob("*.jpg")
    )

    for image_path in image_paths:

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        filename = image_path.name

        for level in levels:

            # Blur
            blur_img = apply_blur(
                image,
                level
            )

            cv2.imwrite(
                str(
                    OUTPUT_ROOT /
                    f"blur_{level}" /
                    filename
                ),
                blur_img
            )

            # Haze
            haze_img = add_haze(
                image,
                level
            )

            cv2.imwrite(
                str(
                    OUTPUT_ROOT /
                    f"haze_{level}" /
                    filename
                ),
                haze_img
            )

            # Blur + Haze
            blur_haze = apply_blur(
                image,
                level
            )

            blur_haze = add_haze(
                blur_haze,
                level
            )

            cv2.imwrite(
                str(
                    OUTPUT_ROOT /
                    f"blur_haze_{level}" /
                    filename
                ),
                blur_haze
            )

    print(
        "Robustness benchmark generated."
    )

if __name__ == "__main__":
    generate_robustness_dataset()
