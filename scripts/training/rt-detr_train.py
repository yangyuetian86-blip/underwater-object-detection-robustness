import os
import shutil
import numpy as np
from pathlib import Path

from ultralytics import RTDETR


def copy_pair(
    file_stem,
    src_img_dir,
    src_label_dir,
    dst_img_dir,
    dst_label_dir
):
    """
    Copy one image and its YOLO-format label.
    """

    src_img = src_img_dir / f"{file_stem}.jpg"
    src_label = src_label_dir / f"{file_stem}.txt"

    if not src_img.exists():
        raise FileNotFoundError(
            f"Missing image: {src_img}"
        )

    if not src_label.exists():
        raise FileNotFoundError(
            f"Missing label: {src_label}"
        )

    shutil.copy(
        src_img,
        dst_img_dir / f"{file_stem}.jpg"
    )

    shutil.copy(
        src_label,
        dst_label_dir / f"{file_stem}.txt"
    )


if __name__ == "__main__":

    # =========================================================
    # Project paths
    # =========================================================

    PROJECT_ROOT = Path(
        r"F:/Program Files/Underwater_Detection"
    )

    DATASET_ROOT = PROJECT_ROOT / "yolo_dataset"

    # Original dataset
    ORIGINAL_IMAGES_DIR = (
        DATASET_ROOT / "images/train"
    )

    ORIGINAL_LABELS_DIR = (
        DATASET_ROOT / "labels/train"
    )

    # Fixed test set
    TEST_IMAGES_DIR = (
        DATASET_ROOT / "images/test"
    )

    TEST_LABELS_DIR = (
        DATASET_ROOT / "labels/test"
    )

    # =========================================================
    # K-fold setting
    # =========================================================

    K = 5

    all_files = sorted([
        f.stem
        for f in ORIGINAL_IMAGES_DIR.glob("*.jpg")
    ])

    test_files = sorted([
        f.stem
        for f in TEST_IMAGES_DIR.glob("*.jpg")
    ])

    print("=" * 70)
    print(f"Train images: {len(all_files)}")
    print(f"Test images : {len(test_files)}")
    print("=" * 70)

    # =========================================================
    # Check labels
    # =========================================================

    for file_name in all_files:

        label_path = (
            ORIGINAL_LABELS_DIR /
            f"{file_name}.txt"
        )

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label: {label_path}"
            )

    for file_name in test_files:

        label_path = (
            TEST_LABELS_DIR /
            f"{file_name}.txt"
        )

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label: {label_path}"
            )

    # =========================================================
    # Check leakage
    # =========================================================

    overlap = set(all_files) & set(test_files)

    if len(overlap) > 0:

        print("Dataset leakage detected.")
        raise ValueError(
            "Train and test overlap."
        )

    else:
        print("No train/test overlap.")

    # =========================================================
    # Shuffle
    # =========================================================

    np.random.seed(42)

    np.random.shuffle(all_files)

    folds = np.array_split(all_files, K)

    # =========================================================
    # Start K-fold training
    # =========================================================

    for i in range(K):

        print("\n" + "=" * 70)
        print(f"Starting Fold {i + 1}/{K}")
        print("=" * 70)

        fold_dir = (
            DATASET_ROOT / f"fold_{i + 1}"
        )

        # Remove old fold
        if fold_dir.exists():
            shutil.rmtree(fold_dir)

        # Create folders
        train_images = (
            fold_dir / "images/train"
        )

        train_labels = (
            fold_dir / "labels/train"
        )

        val_images = (
            fold_dir / "images/val"
        )

        val_labels = (
            fold_dir / "labels/val"
        )

        os.makedirs(train_images, exist_ok=True)
        os.makedirs(train_labels, exist_ok=True)

        os.makedirs(val_images, exist_ok=True)
        os.makedirs(val_labels, exist_ok=True)

        # =====================================================
        # Split data
        # =====================================================

        current_val_files = list(folds[i])

        current_train_files = []

        for j in range(K):

            if j != i:
                current_train_files.extend(
                    folds[j]
                )

        print(f"Train images: {len(current_train_files)}")
        print(f"Val images  : {len(current_val_files)}")

        # =====================================================
        # Copy train set
        # =====================================================

        print("Copying training data...")

        for file_name in current_train_files:

            # Original image
            copy_pair(
                file_name,
                ORIGINAL_IMAGES_DIR,
                ORIGINAL_LABELS_DIR,
                train_images,
                train_labels
            )

        # =====================================================
        # Copy validation set
        # =====================================================

        print("Copying validation data...")

        for file_name in current_val_files:

            copy_pair(
                file_name,
                ORIGINAL_IMAGES_DIR,
                ORIGINAL_LABELS_DIR,
                val_images,
                val_labels
            )

        # =====================================================
        # YAML generation
        # =====================================================

        yaml_content = f"""
            train: {train_images.absolute().as_posix()}
            val: {val_images.absolute().as_posix()}
            test: {TEST_IMAGES_DIR.absolute().as_posix()}

            nc: 4
            names: ['holothurian', 'echinus', 'scallop', 'starfish']
        """

        yaml_path = (
            DATASET_ROOT /
            f"fold_{i + 1}.yaml"
        )

        with open(
            yaml_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(yaml_content.strip())

        print(f"YAML saved: {yaml_path}")

        # =====================================================
        # RT-DETR model
        # =====================================================

        model = RTDETR("rtdetr-l.pt")

        # =====================================================
        # Training
        # =====================================================

        model.train(

            # Dataset
            data=yaml_path.absolute().as_posix(),

            # Training epochs
            epochs=100,

            # Higher resolution for small objects
            imgsz=640,

            # Transformer usually needs smaller batch
            batch=8,

            # Dataloader workers
            workers=4,

            # GPU device
            device=0,

            # Project output
            project=(
                PROJECT_ROOT /
                "runs/rtdetr"
            ).as_posix(),

            # Experiment name
            name=f"rtdetr_fold_{i + 1}",

            exist_ok=True,

            # Cache images
            cache=False,

            # Random seed
            seed=42,

            # Optimizer
            optimizer="AdamW",

            # Initial learning rate
            lr0=1e-4,

            # Weight decay
            weight_decay=1e-4,

            # Early stopping patience
            patience=20,

            # Mixed precision
            amp=True
        )

        print(f"Fold {i + 1} training completed.")

        # =====================================================
        # Test evaluation
        # =====================================================

        best_model_path = (
            PROJECT_ROOT
            / "runs/rtdetr"
            / f"rtdetr_fold_{i + 1}"
            / "weights/best.pt"
        )

        if best_model_path.exists():

            print("Evaluating on test set...")

            best_model = RTDETR(
                best_model_path
            )

            best_model.val(

                data=yaml_path.absolute().as_posix(),

                split="test",

                imgsz=640,

                batch=4,

                workers=2,

                device=0,

                project=(
                    PROJECT_ROOT /
                    "runs/rtdetr"
                ).as_posix(),

                name=f"rtdetr_test_fold_{i + 1}",

                exist_ok=True
            )

            print(
                f"Fold {i + 1} test completed."
            )

        else:

            print(
                f"best.pt not found: {best_model_path}"
            )

    print("\nAll folds completed.")