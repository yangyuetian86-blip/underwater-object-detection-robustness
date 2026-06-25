import os
import shutil
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def copy_pair(file_stem, src_img_dir, src_label_dir, dst_img_dir, dst_label_dir):
    """
    拷贝一张图片和对应 YOLO txt 标签。
    默认图片格式为 .jpg。
    """
    src_img = src_img_dir / f"{file_stem}.jpg"
    src_label = src_label_dir / f"{file_stem}.txt"

    if not src_img.exists():
        raise FileNotFoundError(f"缺失图片: {src_img}")

    if not src_label.exists():
        raise FileNotFoundError(f"缺失标签: {src_label}")

    shutil.copy2(src_img, dst_img_dir / f"{file_stem}.jpg")
    shutil.copy2(src_label, dst_label_dir / f"{file_stem}.txt")


if __name__ == "__main__":

    # 当前项目根目录
    PROJECT_ROOT = Path(r"F:/Program Files/Underwater_Detection")

    # 原始 YOLO 数据集
    ORIGINAL_DATASET_ROOT = PROJECT_ROOT / "yolo_dataset"

    # 增强后的 YOLO 数据集
    AUG_DATASET_ROOT = PROJECT_ROOT / "yolo_dataset_augmented"

    # 原始 trainval 数据，用于划分五折
    ORIGINAL_IMAGES_DIR = ORIGINAL_DATASET_ROOT / "images/train"
    ORIGINAL_LABELS_DIR = ORIGINAL_DATASET_ROOT / "labels/train"

    # 增强数据集中 train 文件夹：里面应该包含 原始图 + _aug 图
    AUG_IMAGES_DIR = AUG_DATASET_ROOT / "images/train"
    AUG_LABELS_DIR = AUG_DATASET_ROOT / "labels/train"

    # 固定 test set，不增强
    TEST_IMAGES_DIR = AUG_DATASET_ROOT / "images/test"
    TEST_LABELS_DIR = AUG_DATASET_ROOT / "labels/test"

    K = 5

    # 只用原始图片名做划分，避免原图和增强图被分到不同 fold 造成数据泄漏
    all_files = sorted([f.stem for f in ORIGINAL_IMAGES_DIR.glob("*.jpg")])
    test_files = sorted([f.stem for f in TEST_IMAGES_DIR.glob("*.jpg")])

    print("=" * 70)
    print(f"原始 trainval 图片数量: {len(all_files)}")
    print(f"固定 test 图片数量: {len(test_files)}")
    print("=" * 70)

    if len(test_files) != 550:
        print(f"⚠️ 注意：当前 test 数量为 {len(test_files)}，不是 550，请确认是否正常。")

    # 检查原始 trainval 标签
    for file_name in all_files:
        label_path = ORIGINAL_LABELS_DIR / f"{file_name}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"原始 trainval 缺失标签: {label_path}")

    # 检查 test 标签
    for file_name in test_files:
        label_path = TEST_LABELS_DIR / f"{file_name}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"test 缺失标签: {label_path}")

    # 检查 test 是否混入 trainval
    overlap = set(all_files) & set(test_files)
    if len(overlap) > 0:
        print("❌ 发现 trainval 和 test 存在重名图片：")
        print(list(overlap)[:20])
        raise ValueError("test 集与 trainval 集存在数据泄漏，请先重新划分。")
    else:
        print("✅ trainval 与 test 文件名无重叠。")

    # 固定随机种子
    np.random.seed(42)
    np.random.shuffle(all_files)

    folds = np.array_split(all_files, K)

    for i in range(2,K):
        print("\n" + "=" * 70)
        print(f"🚀 正在开始第 {i + 1}/{K} 折增强训练...")
        print("=" * 70)

        fold_dir = AUG_DATASET_ROOT / f"fold_{i + 1}"

        # 每次重新生成 fold，避免旧数据残留
        if fold_dir.exists():
            shutil.rmtree(fold_dir)

        train_images = fold_dir / "images/train"
        train_labels = fold_dir / "labels/train"
        val_images = fold_dir / "images/val"
        val_labels = fold_dir / "labels/val"

        os.makedirs(train_images, exist_ok=True)
        os.makedirs(train_labels, exist_ok=True)
        os.makedirs(val_images, exist_ok=True)
        os.makedirs(val_labels, exist_ok=True)

        current_val_files = list(folds[i])

        current_train_files = []
        for j in range(K):
            if j != i:
                current_train_files.extend(folds[j])

        print(f"📁 第 {i + 1} 折数据分配：")
        print(f"   原始 train: {len(current_train_files)} 张")
        print(f"   原始 val:   {len(current_val_files)} 张")
        print(f"   test:       {len(test_files)} 张，固定不参与训练")
        print("⏳ 正在拷贝 train/val 文件...")

        # 训练集：原始图 + 对应增强图
        aug_used_count = 0

        for file_name in current_train_files:
            # 复制原始图
            copy_pair(
                file_name,
                ORIGINAL_IMAGES_DIR,
                ORIGINAL_LABELS_DIR,
                train_images,
                train_labels
            )

            # 如果存在增强图，则也加入训练集
            aug_name = f"{file_name}_aug"
            aug_img_path = AUG_IMAGES_DIR / f"{aug_name}.jpg"
            aug_label_path = AUG_LABELS_DIR / f"{aug_name}.txt"

            if aug_img_path.exists() and aug_label_path.exists():
                copy_pair(
                    aug_name,
                    AUG_IMAGES_DIR,
                    AUG_LABELS_DIR,
                    train_images,
                    train_labels
                )
                aug_used_count += 1

        # 验证集：只放原始图，不放增强图
        for file_name in current_val_files:
            copy_pair(
                file_name,
                ORIGINAL_IMAGES_DIR,
                ORIGINAL_LABELS_DIR,
                val_images,
                val_labels
            )

        print(f"✅ 第 {i + 1} 折实际训练集数量: {len(list(train_images.glob('*.jpg')))} 张")
        print(f"   其中增强图数量: {aug_used_count} 张")
        print(f"✅ 第 {i + 1} 折验证集数量: {len(list(val_images.glob('*.jpg')))} 张")

        yaml_content = f"""
train: {train_images.absolute().as_posix()}
val: {val_images.absolute().as_posix()}
test: {TEST_IMAGES_DIR.absolute().as_posix()}

nc: 4
names: ['holothurian', 'echinus', 'scallop', 'starfish']
"""

        yaml_path = AUG_DATASET_ROOT / f"fold_{i + 1}.yaml"

        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content.strip())

        print(f"✅ 第 {i + 1} 折 YAML 已生成: {yaml_path}")

        # 每一折都从 YOLO11s 预训练权重重新开始
        model = YOLO(PROJECT_ROOT / "yolo11s.pt")

        model.train(
            data=yaml_path.absolute().as_posix(),
            epochs=100,
            imgsz=640,
            batch=24,
            workers=4,
            device=0,
            close_mosaic=10,
            project=(PROJECT_ROOT / "runs/detect").as_posix(),
            name=f"yolo11s_aug_train_fold_{i + 1}",
            exist_ok=True,
            cache=False,
            seed=42
        )

        print(f"✅ 第 {i + 1} 折增强训练完成。")

        # 当前 fold 的 best.pt
        best_model_path = (
            PROJECT_ROOT
            / "runs/detect"
            / f"yolo11s_aug_train_fold_{i + 1}"
            / "weights/best.pt"
        )

        if best_model_path.exists():
            print(f"🧪 正在用第 {i + 1} 折 best.pt 测试固定 test 集...")

            best_model = YOLO(best_model_path)

            best_model.val(
                data=yaml_path.absolute().as_posix(),
                split="test",
                imgsz=640,
                batch=8,
                workers=2,
                device=0,
                project=(PROJECT_ROOT / "runs/detect").as_posix(),
                name=f"yolo11s_aug_test_fold_{i + 1}",
                exist_ok=True
            )

            print(f"✅ 第 {i + 1} 折增强模型 test 测试完成。")
        else:
            print(f"⚠️ 没有找到 best.pt: {best_model_path}")

    print("\n🎉 YOLO11s 增强数据五折训练和固定 test 测试全部完成！")