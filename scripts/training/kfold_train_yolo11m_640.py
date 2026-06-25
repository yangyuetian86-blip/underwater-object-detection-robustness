import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO


def copy_pair(file_stem, src_img_dir, src_label_dir, dst_img_dir, dst_label_dir):
    """
    拷贝一张图片和对应的 YOLO txt 标签。
    默认图片格式为 .jpg。
    """
    src_img = src_img_dir / f"{file_stem}.jpg"
    src_label = src_label_dir / f"{file_stem}.txt"

    if not src_img.exists():
        raise FileNotFoundError(f"缺失图片: {src_img}")

    if not src_label.exists():
        raise FileNotFoundError(f"缺失标签: {src_label}")

    shutil.copy(src_img, dst_img_dir / f"{file_stem}.jpg")
    shutil.copy(src_label, dst_label_dir / f"{file_stem}.txt")


def extract_metrics(metrics):
    """
    提取 Ultralytics 验证结果中的整体指标。
    """
    return {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
    }


if __name__ == "__main__":

    # =========================
    # 1. 路径设置
    # =========================

    PROJECT_ROOT = Path(r"F:/Program Files/Underwater_Detection")

    # 原始 YOLO 数据集根目录
    DATASET_ROOT = PROJECT_ROOT / "yolo_dataset"

    # YOLO11m 640 的五折临时数据目录
    # 不放到 yolo_dataset 里面，避免污染原始数据
    KFOLD_ROOT = PROJECT_ROOT / "kfold_yolo11m_640"
    KFOLD_ROOT.mkdir(parents=True, exist_ok=True)

    # 训练结果保存目录
    RUNS_DIR = PROJECT_ROOT / "runs/detect"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # 原始 trainval 数据集
    ORIGINAL_IMAGES_DIR = DATASET_ROOT / "images/train"
    ORIGINAL_LABELS_DIR = DATASET_ROOT / "labels/train"

    # 固定 test 数据集
    TEST_IMAGES_DIR = DATASET_ROOT / "images/test"
    TEST_LABELS_DIR = DATASET_ROOT / "labels/test"

    # =========================
    # 2. 训练参数
    # =========================

    K = 5
    SEED = 42

    EPOCHS = 100
    IMGSZ = 640
    BATCH = 12
    WORKERS = 2
    DEVICE = 0

    CLASS_NAMES = ['holothurian', 'echinus', 'scallop', 'starfish']

    # =========================
    # 3. 读取图片文件
    # =========================

    all_files = sorted([
        f.stem
        for f in ORIGINAL_IMAGES_DIR.glob("*.jpg")
    ])

    test_files = sorted([
        f.stem
        for f in TEST_IMAGES_DIR.glob("*.jpg")
    ])

    print("=" * 70)
    print("YOLO11m 640 五折交叉验证训练")
    print("=" * 70)
    print(f"trainval 图片数量: {len(all_files)}")
    print(f"test 图片数量: {len(test_files)}")
    print("=" * 70)

    if len(all_files) == 0:
        raise RuntimeError("❌ 没有找到 trainval 图片，请检查 yolo_dataset/images/train")

    if len(test_files) == 0:
        raise RuntimeError("❌ 没有找到 test 图片，请检查 yolo_dataset/images/test")

    if len(all_files) != 4993 and len(all_files) != 4905:
        print(f"⚠️ 注意：当前 trainval 数量为 {len(all_files)}，请确认是否正常。")

    if len(test_files) != 550:
        print(f"⚠️ 注意：当前 test 数量为 {len(test_files)}，不是 550，请确认是否正常。")

    # =========================
    # 4. 检查标签完整性
    # =========================

    for file_name in all_files:
        label_path = ORIGINAL_LABELS_DIR / f"{file_name}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"trainval 缺失标签: {label_path}")

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

    # =========================
    # 5. K 折划分
    # =========================

    np.random.seed(SEED)
    np.random.shuffle(all_files)

    folds = np.array_split(all_files, K)

    test_summary_rows = []
    val_summary_rows = []

    # =========================
    # 6. 开始五折训练
    # =========================

    for i in range(K):
        fold_num = i + 1

        # 只跑第 1 折
       # if fold_num != 1:
       #     continue

        print("\n" + "=" * 70)
        print(f"🚀 正在开始第 {fold_num}/{K} 折训练...")
        print("=" * 70)

        # 当前 fold 的目录
        fold_dir = KFOLD_ROOT / f"fold_{fold_num}"

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

        print(f"📁 第 {fold_num} 折数据分配：")
        print(f"   train: {len(current_train_files)} 张")
        print(f"   val:   {len(current_val_files)} 张")
        print(f"   test:  {len(test_files)} 张，固定不参与训练")
        print("⏳ 正在拷贝 train/val 文件...")

        # 复制 train 数据
        for file_name in current_train_files:
            copy_pair(
                file_name,
                ORIGINAL_IMAGES_DIR,
                ORIGINAL_LABELS_DIR,
                train_images,
                train_labels
            )

        # 复制 val 数据
        for file_name in current_val_files:
            copy_pair(
                file_name,
                ORIGINAL_IMAGES_DIR,
                ORIGINAL_LABELS_DIR,
                val_images,
                val_labels
            )

        train_count = len(list(train_images.glob("*.jpg")))
        val_count = len(list(val_images.glob("*.jpg")))

        print(f"✅ 实际 train 图片数量: {train_count}")
        print(f"✅ 实际 val 图片数量: {val_count}")

        if train_count == 0:
            raise RuntimeError("❌ train 图片数量为 0，说明复制失败。")

        if val_count == 0:
            raise RuntimeError("❌ val 图片数量为 0，说明复制失败。")

        # =========================
        # 7. 生成 YAML
        # =========================

        yaml_content = (
            f'train: "{train_images.absolute().as_posix()}"\n'
            f'val: "{val_images.absolute().as_posix()}"\n'
            f'test: "{TEST_IMAGES_DIR.absolute().as_posix()}"\n'
            f'\n'
            f'nc: 4\n'
            f"names: {CLASS_NAMES}\n"
        )

        yaml_path = fold_dir / f"fold_{fold_num}.yaml"

        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content.strip())

        print(f"✅ 第 {fold_num} 折 YAML 已生成: {yaml_path}")
        print("当前 YAML 内容：")
        print(yaml_content)

        # =========================
        # 8. 训练 YOLO11m 640
        # =========================

        print(f"\n🚀 开始训练 YOLO11m 640 Fold {fold_num}...")

        model = YOLO("yolo11m.pt")

        model.train(
            data=yaml_path.absolute().as_posix(),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            workers=WORKERS,
            device=DEVICE,
            close_mosaic=10,
            project=RUNS_DIR.as_posix(),
            name=f"yolo11m_640_train_fold_{fold_num}",
            exist_ok=True,
            cache=False,
            seed=SEED
        )

        print(f"✅ 第 {fold_num} 折训练完成。")

        # =========================
        # 9. 加载 best.pt
        # =========================

        best_model_path = RUNS_DIR / f"yolo11m_640_train_fold_{fold_num}" / "weights" / "best.pt"

        if not best_model_path.exists():
            print(f"⚠️ 没有找到 best.pt: {best_model_path}")
            continue

        print(f"🧪 正在加载第 {fold_num} 折 best.pt:")
        print(best_model_path)

        best_model = YOLO(best_model_path.as_posix())

        # =========================
        # 10. val 集评估
        # =========================

        print(f"\n📊 正在测试第 {fold_num} 折 val 集...")

        val_metrics = best_model.val(
            data=yaml_path.absolute().as_posix(),
            split="val",
            imgsz=IMGSZ,
            batch=BATCH,
            workers=0,
            device=DEVICE,
            project=RUNS_DIR.as_posix(),
            name=f"yolo11m_640_val_fold_{fold_num}",
            exist_ok=True
        )

        val_row = {
            "fold": fold_num,
            **extract_metrics(val_metrics)
        }

        val_summary_rows.append(val_row)

        print(f"✅ 第 {fold_num} 折 val 结果：")
        print(val_row)

        # =========================
        # 11. 固定 test 集评估
        # =========================

        print(f"\n📊 正在用第 {fold_num} 折 best.pt 测试固定 test 集...")

        test_metrics = best_model.val(
            data=yaml_path.absolute().as_posix(),
            split="test",
            imgsz=IMGSZ,
            batch=BATCH,
            workers=0,
            device=DEVICE,
            project=RUNS_DIR.as_posix(),
            name=f"yolo11m_640_test_fold_{fold_num}",
            exist_ok=True
        )

        test_row = {
            "fold": fold_num,
            **extract_metrics(test_metrics)
        }

        test_summary_rows.append(test_row)

        print(f"✅ 第 {fold_num} 折 test 结果：")
        print(test_row)

        # =========================
        # 12. 输出每类别 test 结果
        # =========================

        print(f"\n========== YOLO11m 640 Fold {fold_num} Test 每类别结果 ==========")

        class_names = {
            0: "holothurian",
            1: "echinus",
            2: "scallop",
            3: "starfish"
        }

        for cls_id, cls_name in class_names.items():
            p, r, ap50, ap = test_metrics.class_result(cls_id)

            print(f"\nClass {cls_id}: {cls_name}")
            print(f"Precision : {p:.4f}")
            print(f"Recall    : {r:.4f}")
            print(f"mAP50     : {ap50:.4f}")
            print(f"mAP50-95  : {ap:.4f}")

        print(f"\n✅ 第 {fold_num} 折 test 测试完成。")

    # =========================
    # 13. 保存 test summary
    # =========================

    if len(test_summary_rows) > 0:
        test_df = pd.DataFrame(test_summary_rows)

        mean_row = {
            "fold": "mean",
            "precision": test_df["precision"].mean(),
            "recall": test_df["recall"].mean(),
            "mAP50": test_df["mAP50"].mean(),
            "mAP50-95": test_df["mAP50-95"].mean(),
        }

        std_row = {
            "fold": "std",
            "precision": test_df["precision"].std(),
            "recall": test_df["recall"].std(),
            "mAP50": test_df["mAP50"].std(),
            "mAP50-95": test_df["mAP50-95"].std(),
        }

        test_df = pd.concat(
            [test_df, pd.DataFrame([mean_row, std_row])],
            ignore_index=True
        )

        test_save_path = RUNS_DIR / "yolo11m_640_kfold_test_summary.csv"
        test_df.to_csv(test_save_path, index=False, encoding="utf-8-sig")

        print(f"\n✅ test summary 已保存: {test_save_path}")
        print(test_df)

    # =========================
    # 14. 保存 val summary
    # =========================

    if len(val_summary_rows) > 0:
        val_df = pd.DataFrame(val_summary_rows)

        mean_row = {
            "fold": "mean",
            "precision": val_df["precision"].mean(),
            "recall": val_df["recall"].mean(),
            "mAP50": val_df["mAP50"].mean(),
            "mAP50-95": val_df["mAP50-95"].mean(),
        }

        std_row = {
            "fold": "std",
            "precision": val_df["precision"].std(),
            "recall": val_df["recall"].std(),
            "mAP50": val_df["mAP50"].std(),
            "mAP50-95": val_df["mAP50-95"].std(),
        }

        val_df = pd.concat(
            [val_df, pd.DataFrame([mean_row, std_row])],
            ignore_index=True
        )

        val_save_path = RUNS_DIR / "yolo11m_640_kfold_val_summary.csv"
        val_df.to_csv(val_save_path, index=False, encoding="utf-8-sig")

        print(f"\n✅ val summary 已保存: {val_save_path}")
        print(val_df)

    print("\n🎉 YOLO11m 640 五折交叉验证训练和固定 test 测试全部完成！")