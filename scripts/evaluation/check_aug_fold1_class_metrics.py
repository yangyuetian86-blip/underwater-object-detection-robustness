from pathlib import Path
from ultralytics import YOLO


def main():
    PROJECT_ROOT = Path(r"F:/Program Files/Underwater_Detection")

    ARCHIVE_ROOT = PROJECT_ROOT / "Archive_YOLO11s_Aug_5Fold_Results"

    MODEL_PATH = ARCHIVE_ROOT / "yolo11s_aug_train_fold_1/weights/best.pt"
    YAML_PATH = ARCHIVE_ROOT / "fold_yaml/fold_1.yaml"

    print("模型路径:", MODEL_PATH)
    print("YAML路径:", YAML_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型: {MODEL_PATH}")

    if not YAML_PATH.exists():
        raise FileNotFoundError(f"找不到 YAML: {YAML_PATH}")

    model = YOLO(MODEL_PATH.as_posix())

    metrics = model.val(
        data=YAML_PATH.as_posix(),
        split="test",
        imgsz=640,   # 普通全图增强实验如果当时是640，就保持640
        batch=8,
        workers=0,   # Windows 下建议用 0，避免 multiprocessing 报错
        device=0,
        project=(PROJECT_ROOT / "runs/detect").as_posix(),
        name="check_aug_fold1_class_metrics",
        exist_ok=True
    )

    print("\n========== YOLO11s 全图增强 aug fold 1 每类别结果 ==========")

    class_names = {
        0: "holothurian",
        1: "echinus",
        2: "scallop",
        3: "starfish"
    }

    for cls_id, cls_name in class_names.items():
        p, r, ap50, ap = metrics.class_result(cls_id)

        print(f"\nClass {cls_id}: {cls_name}")
        print(f"Precision : {p:.4f}")
        print(f"Recall    : {r:.4f}")
        print(f"mAP50     : {ap50:.4f}")
        print(f"mAP50-95  : {ap:.4f}")


if __name__ == "__main__":
    main()