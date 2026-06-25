import os
import shutil
import numpy as np
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


if __name__ == "__main__":

    # 当前项目根目录
    PROJECT_ROOT = Path(r"F:/Program Files/Underwater_Detection")

    # 原始数据集：只用于五折划分，避免原图和增强图被分到不同 fold
    ORIGINAL_DATASET_ROOT = PROJECT_ROOT / "yolo_dataset"

    # 类别定向增强数据集：里面包含 原图 + _clsaug 增强图
    CLS_AUG_DATASET_ROOT = PROJECT_ROOT / "yolo_dataset_clsaug"

    # fold 文件和 yaml 保存到增强数据集目录
    DATASET_ROOT = CLS_AUG_DATASET_ROOT

    # 原始 trainval 数据，只用它来做 5 折划分
    ORIGINAL_IMAGES_DIR = ORIGINAL_DATASET_ROOT / "images/train"
    ORIGINAL_LABELS_DIR = ORIGINAL_DATASET_ROOT / "labels/train"

    # 类别定向增强后的 train 数据，用来找 _clsaug_1 / _clsaug_2
    CLS_AUG_IMAGES_DIR = CLS_AUG_DATASET_ROOT / "images/train"
    CLS_AUG_LABELS_DIR = CLS_AUG_DATASET_ROOT / "labels/train"

    # 固定 test 集，不增强
    TEST_IMAGES_DIR = CLS_AUG_DATASET_ROOT / "images/test"
    TEST_LABELS_DIR = CLS_AUG_DATASET_ROOT / "labels/test"

    K = 5

    all_files = sorted([
        f.stem
        for f in ORIGINAL_IMAGES_DIR.glob("*.jpg")
    ])
    test_files = sorted([f.stem for f in TEST_IMAGES_DIR.glob("*.jpg")])

    print("=" * 70)
    print(f"trainval 图片数量: {len(all_files)}")
    print(f"test 图片数量: {len(test_files)}")
    print("=" * 70)

    if len(all_files) != 4993:
        print(f"⚠️ 注意：当前 trainval 数量为 {len(all_files)}，不是 4993，请确认是否正常。")

    if len(test_files) != 550:
        print(f"⚠️ 注意：当前 test 数量为 {len(test_files)}，不是 550，请确认是否正常。")

    # 检查 trainval 标签是否完整
    for file_name in all_files:
        label_path = ORIGINAL_LABELS_DIR / f"{file_name}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"trainval 缺失标签: {label_path}")

    # 检查 test 标签是否完整
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

    for i in range(1):
        print("\n" + "=" * 70)
        print(f"🚀 正在开始第 {i + 1}/{K} 折训练...")
        print("=" * 70)

        fold_dir = DATASET_ROOT / f"fold_{i + 1}"

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
        print(f"   train: {len(current_train_files)} 张")
        print(f"   val:   {len(current_val_files)} 张")
        print(f"   test:  {len(test_files)} 张，固定不参与训练")
        print("⏳ 正在拷贝 train/val 文件...")

        clsaug_used_count = 0

        for file_name in current_train_files:
            copy_pair(
                file_name,
                ORIGINAL_IMAGES_DIR,
                ORIGINAL_LABELS_DIR,
                train_images,
                train_labels
            )

            # 只把当前训练折对应的类别增强图加入 train，避免数据泄漏
            for aug_idx in range(1, 3):
                aug_name = f"{file_name}_clsaug_{aug_idx}"

                aug_img_path = CLS_AUG_IMAGES_DIR / f"{aug_name}.jpg"
                aug_label_path = CLS_AUG_LABELS_DIR / f"{aug_name}.txt"

                if aug_img_path.exists() and aug_label_path.exists():
                    copy_pair(
                        aug_name,
                        CLS_AUG_IMAGES_DIR,
                        CLS_AUG_LABELS_DIR,
                        train_images,
                        train_labels
                    )
                    
                    clsaug_used_count += 1

            # 肖云天：同时拷贝增强数据
            # 此时训练集包含原始图和增强图两部分，共3994+(增强后图片数量?)=(?)，验证集只包含原始图999张，测试集保持不变550张
            #copy_pair(
            #    f"{file_name}_aug",
            #    AUG_IMAGES_DIR,
            #    AUG_LABELS_DIR,
            #    train_images,
            #    train_labels
            #)

        for file_name in current_val_files:
            copy_pair(
                file_name,
                ORIGINAL_IMAGES_DIR,
                ORIGINAL_LABELS_DIR,
                val_images,
                val_labels
            )

        print(f"✅ 实际 train 图片数量: {len(list(train_images.glob('*.jpg')))}")
        print(f"   其中类别增强图数量: {clsaug_used_count}")
        print(f"✅ 实际 val 图片数量: {len(list(val_images.glob('*.jpg')))}")

        yaml_content = (
            f'train: "{train_images.absolute().as_posix()}"\n'
            f'val: "{val_images.absolute().as_posix()}"\n'
            f'test: "{TEST_IMAGES_DIR.absolute().as_posix()}"\n'
            f'\n'
            f'nc: 4\n'
            f"names: ['holothurian', 'echinus', 'scallop', 'starfish']\n"
        )

        yaml_path = DATASET_ROOT / f"fold_{i + 1}.yaml"

        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content.strip())

        print(f"✅ 第 {i + 1} 折 YAML 已生成: {yaml_path}")

        # 每一折都从相同预训练权重重新开始
        #model = YOLO(PROJECT_ROOT / "yolo11n.pt")
        model = YOLO("yolo11s.pt")

        model.train(
            data=yaml_path.absolute().as_posix(),
            epochs=100, # 肖云天：early stopping，减少到75轮，观察是否过拟合（有连续10个epoch准确率下降或不变），如果过拟合严重，可以进一步减少到50轮
            imgsz=768,
            batch=16, # 肖云天：增强后训练集增大，适当增大batch size以充分利用显存，如果显存不足可以减小到16
            workers=4, # 肖云天：根据显存调整workers数量，如果有12G，可以尝试8
            device=0,
            close_mosaic=10,
            project=(PROJECT_ROOT / "runs/detect").as_posix(),
            name=f"yolo11s_768_clsaug_train_fold_{i + 1}",
            exist_ok=True,
            cache=False,
            seed=42
        )

        print(f"✅ 第 {i + 1} 折训练完成。")

        # 当前 fold 的 best.pt
        best_model_path = PROJECT_ROOT / "runs/detect" / f"yolo11s_768_clsaug_train_fold_{i + 1}" / "weights/best.pt"

        if best_model_path.exists():
            print(f"🧪 正在用第 {i + 1} 折 best.pt 测试固定 test 集...")

            best_model = YOLO(best_model_path)

            best_model.val(
                data=yaml_path.absolute().as_posix(),
                split="test",
                imgsz=768,
                batch=16,
                workers=2,
                device=0,
                project=(PROJECT_ROOT / "runs/detect").as_posix(),
                name=f"yolo11s_768_clsaug_test_fold_{i + 1}",
                exist_ok=True
            )

            print(f"✅ 第 {i + 1} 折 test 测试完成。")
        else:
            print(f"⚠️ 没有找到 best.pt: {best_model_path}")

    print("\n🎉 5 折交叉验证训练和固定 test 测试全部完成！")