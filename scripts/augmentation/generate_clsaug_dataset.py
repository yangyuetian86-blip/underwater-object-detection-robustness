from pathlib import Path
import shutil
import random

import cv2
import numpy as np


# ============================================================
# 路径设置
# ============================================================
PROJECT_ROOT = Path(r"F:/Program Files/Underwater_Detection")

ORIGINAL_DATASET_ROOT = PROJECT_ROOT / "yolo_dataset"
CLS_AUG_DATASET_ROOT = PROJECT_ROOT / "yolo_dataset_clsaug"

# 原始 train 数据
INPUT_IMAGE_DIR = ORIGINAL_DATASET_ROOT / "images/train"
INPUT_LABEL_DIR = ORIGINAL_DATASET_ROOT / "labels/train"

# 原始 test 数据，不增强，只复制
TEST_IMAGE_DIR = ORIGINAL_DATASET_ROOT / "images/test"
TEST_LABEL_DIR = ORIGINAL_DATASET_ROOT / "labels/test"

# 新数据集输出路径
OUTPUT_TRAIN_IMAGE_DIR = CLS_AUG_DATASET_ROOT / "images/train"
OUTPUT_TRAIN_LABEL_DIR = CLS_AUG_DATASET_ROOT / "labels/train"

OUTPUT_TEST_IMAGE_DIR = CLS_AUG_DATASET_ROOT / "images/test"
OUTPUT_TEST_LABEL_DIR = CLS_AUG_DATASET_ROOT / "labels/test"


# ============================================================
# 类别说明
# 0 = holothurian
# 1 = echinus
# 2 = scallop
# 3 = starfish
# ============================================================


def get_classes_in_label(label_path):
    """
    读取一张图片对应的 YOLO 标签，返回其中包含的类别集合。
    """
    classes = set()

    if not label_path.exists():
        return classes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            cls_id = int(float(parts[0]))
            classes.add(cls_id)

    return classes


# ============================================================
# 温和增强函数
# ============================================================

def mild_clahe(image):
    """
    轻微 CLAHE：增强亮度通道对比度，但不要太强。
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def gamma_correction(image, gamma=0.9):
    """
    Gamma 校正。
    gamma < 1 会轻微提亮暗部，有利于低对比度 holothurian。
    """
    img = image.astype(np.float32) / 255.0
    img = np.power(img, gamma)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    return img


def mild_sharpen(image):
    """
    轻微锐化，增强边缘。
    对小 scallop 边界有一定帮助。
    """
    blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
    sharpened = cv2.addWeighted(image, 1.3, blurred, -0.3, 0)

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def mild_brightness_contrast(image):
    """
    轻微亮度和对比度扰动。
    不做过强变化，避免训练分布偏移太大。
    """
    alpha = random.uniform(1.03, 1.12)  # contrast
    beta = random.uniform(-5, 8)        # brightness

    output = cv2.convertScaleAbs(
        image,
        alpha=alpha,
        beta=beta
    )

    return output


def class_aware_pipeline(image, classes):
    """
    类别感知增强：
    - holothurian: 低对比、伪装，重点做温和 CLAHE / gamma / 对比度增强
    - scallop: 小目标，重点保留边界，做轻微锐化 / 对比度增强
    - 其他类别：不重点增强
    """

    output = image.copy()

    has_holothurian = 0 in classes
    has_scallop = 2 in classes

    if has_holothurian and has_scallop:
        # 同时包含 holothurian 和 scallop
        output = mild_clahe(output)

        if random.random() < 0.7:
            output = gamma_correction(
                output,
                gamma=random.uniform(0.85, 0.95)
            )

        if random.random() < 0.6:
            output = mild_sharpen(output)

        if random.random() < 0.5:
            output = mild_brightness_contrast(output)

    elif has_holothurian:
        # holothurian：提升低对比目标可见性
        output = mild_clahe(output)

        if random.random() < 0.8:
            output = gamma_correction(
                output,
                gamma=random.uniform(0.85, 0.95)
            )

        if random.random() < 0.5:
            output = mild_brightness_contrast(output)

        if random.random() < 0.4:
            output = mild_sharpen(output)

    elif has_scallop:
        # scallop：保留小目标边界
        if random.random() < 0.7:
            output = mild_sharpen(output)

        if random.random() < 0.6:
            output = mild_brightness_contrast(output)

        if random.random() < 0.4:
            output = mild_clahe(output)

    else:
        # 其他类别不增强，避免 echinus 进一步占优势
        return None

    return output


# ============================================================
# 主函数：生成类别感知增强数据集
# ============================================================

def generate_clsaug_dataset():
    print("=" * 70)
    print("开始生成类别感知增强数据集 yolo_dataset_clsaug")
    print("=" * 70)

    # 如果旧数据集存在，先删除，避免旧文件混入
    if CLS_AUG_DATASET_ROOT.exists():
        print(f"删除旧数据集: {CLS_AUG_DATASET_ROOT}")
        shutil.rmtree(CLS_AUG_DATASET_ROOT)

    # 创建输出文件夹
    for p in [
        OUTPUT_TRAIN_IMAGE_DIR,
        OUTPUT_TRAIN_LABEL_DIR,
        OUTPUT_TEST_IMAGE_DIR,
        OUTPUT_TEST_LABEL_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1. 复制原始 train 数据
    # ------------------------------------------------------------
    print("正在复制原始 train 数据...")

    train_images = sorted(INPUT_IMAGE_DIR.glob("*.jpg"))

    original_train_count = 0

    for img_path in train_images:
        label_path = INPUT_LABEL_DIR / f"{img_path.stem}.txt"

        if not label_path.exists():
            print(f"缺失 train 标签，跳过: {label_path}")
            continue

        shutil.copy2(
            img_path,
            OUTPUT_TRAIN_IMAGE_DIR / img_path.name
        )

        shutil.copy2(
            label_path,
            OUTPUT_TRAIN_LABEL_DIR / label_path.name
        )

        original_train_count += 1

    print(f"原始 train 复制完成: {original_train_count} 张")

    # ------------------------------------------------------------
    # 2. 复制原始 test 数据
    # ------------------------------------------------------------
    print("正在复制固定 test 数据...")

    test_images = sorted(TEST_IMAGE_DIR.glob("*.jpg"))

    test_count = 0

    for img_path in test_images:
        label_path = TEST_LABEL_DIR / f"{img_path.stem}.txt"

        if not label_path.exists():
            print(f"缺失 test 标签，跳过: {label_path}")
            continue

        shutil.copy2(
            img_path,
            OUTPUT_TEST_IMAGE_DIR / img_path.name
        )

        shutil.copy2(
            label_path,
            OUTPUT_TEST_LABEL_DIR / label_path.name
        )

        test_count += 1

    print(f"固定 test 复制完成: {test_count} 张")

    # ------------------------------------------------------------
    # 3. 对 holothurian / scallop 做类别定向增强
    # ------------------------------------------------------------
    print("正在生成类别定向增强图像...")

    clsaug_count = 0
    skipped_count = 0

    for idx, img_path in enumerate(train_images):
        label_path = INPUT_LABEL_DIR / f"{img_path.stem}.txt"

        if not label_path.exists():
            continue

        classes = get_classes_in_label(label_path)

        # 只增强包含 holothurian 或 scallop 的图片
        if 0 in classes or 2 in classes:
            augment_times = 2
        else:
            augment_times = 0

        if augment_times == 0:
            skipped_count += 1
            continue

        image = cv2.imread(str(img_path))

        if image is None:
            print(f"无法读取图片，跳过: {img_path}")
            continue

        stem = img_path.stem

        for aug_idx in range(augment_times):
            augmented = class_aware_pipeline(image, classes)

            if augmented is None:
                continue

            aug_img_name = f"{stem}_clsaug_{aug_idx + 1}.jpg"
            aug_label_name = f"{stem}_clsaug_{aug_idx + 1}.txt"

            cv2.imwrite(
                str(OUTPUT_TRAIN_IMAGE_DIR / aug_img_name),
                augmented
            )

            shutil.copy2(
                label_path,
                OUTPUT_TRAIN_LABEL_DIR / aug_label_name
            )

            clsaug_count += 1

        if idx % 100 == 0:
            print(f"[{idx}/{len(train_images)}] processed...")

    print("=" * 70)
    print("类别感知增强数据集生成完成")
    print("=" * 70)
    print(f"原始 train 数量: {original_train_count}")
    print(f"增强 train 数量: {clsaug_count}")
    print(f"跳过非目标类别图片数量: {skipped_count}")
    print(f"固定 test 数量: {test_count}")
    print(f"输出位置: {CLS_AUG_DATASET_ROOT}")


if __name__ == "__main__":
    generate_clsaug_dataset()