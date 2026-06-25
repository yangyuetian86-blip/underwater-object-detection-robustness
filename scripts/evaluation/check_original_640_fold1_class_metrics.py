from pathlib import Path
from ultralytics import YOLO


def main():
    PROJECT_ROOT = Path(r"F:/Program Files/Underwater_Detection")

    MODEL_PATH = PROJECT_ROOT / "runs/detect/yolo11s_train_fold_1/weights/best.pt"
    YAML_PATH = PROJECT_ROOT / "original_test.yaml"

    yaml_content = (
        f'train: "{(PROJECT_ROOT / "yolo_dataset/images/train").as_posix()}"\n'
        f'val: "{(PROJECT_ROOT / "yolo_dataset/images/train").as_posix()}"\n'
        f'test: "{(PROJECT_ROOT / "yolo_dataset/images/test").as_posix()}"\n'
        f'\n'
        f'nc: 4\n'
        f"names: ['holothurian', 'echinus', 'scallop', 'starfish']\n"
    )

    with open(YAML_PATH, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print("模型路径:", MODEL_PATH)
    print("YAML路径:", YAML_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型: {MODEL_PATH}")

    model = YOLO(MODEL_PATH.as_posix())

    metrics = model.val(
        data=YAML_PATH.as_posix(),
        split="test",
        imgsz=640,
        batch=8,
        workers=0,
        device=0,
        project=(PROJECT_ROOT / "runs/detect").as_posix(),
        name="check_original_640_fold1_class_metrics",
        exist_ok=True
    )

    print("\n========== 原始 YOLO11s 640 fold 1 每类别结果 ==========")

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