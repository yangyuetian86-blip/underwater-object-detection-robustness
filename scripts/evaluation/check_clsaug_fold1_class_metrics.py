from pathlib import Path
import csv
from ultralytics import YOLO


def main():
    PROJECT_ROOT = Path(r"F:/Program Files/Underwater_Detection")

    MODEL_PATH = PROJECT_ROOT / "runs/detect/yolo11s_768_clsaug_train_fold_1/weights/best.pt"
    YAML_PATH = PROJECT_ROOT / "yolo_dataset_clsaug/fold_1.yaml"

    SAVE_CSV = PROJECT_ROOT / "clsaug_fold1_class_metrics.csv"

    print("Model path:", MODEL_PATH)
    print("YAML path:", YAML_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not YAML_PATH.exists():
        raise FileNotFoundError(f"YAML not found: {YAML_PATH}")

    model = YOLO(MODEL_PATH.as_posix())

    metrics = model.val(
        data=YAML_PATH.as_posix(),
        split="test",
        imgsz=768,
        batch=8,
        workers=0,
        device=0,
        project=(PROJECT_ROOT / "runs/detect").as_posix(),
        name="check_clsaug_fold1_class_metrics",
        exist_ok=True
    )

    class_names = {
        0: "holothurian",
        1: "echinus",
        2: "scallop",
        3: "starfish"
    }

    rows = []

    print("\n========== YOLO11s 768 class-aware augmentation fold 1 per-class results ==========")

    for cls_id, cls_name in class_names.items():
        p, r, ap50, ap = metrics.class_result(cls_id)

        print(f"\nClass {cls_id}: {cls_name}")
        print(f"Precision : {p:.4f}")
        print(f"Recall    : {r:.4f}")
        print(f"mAP50     : {ap50:.4f}")
        print(f"mAP50-95  : {ap:.4f}")

        rows.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "precision": round(p, 6),
            "recall": round(r, 6),
            "mAP50": round(ap50, 6),
            "mAP50-95": round(ap, 6)
        })

    with open(SAVE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_id", "class_name", "precision", "recall", "mAP50", "mAP50-95"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved per-class results to: {SAVE_CSV}")


if __name__ == "__main__":
    main()