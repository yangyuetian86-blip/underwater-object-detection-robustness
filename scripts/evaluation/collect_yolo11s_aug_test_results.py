from pathlib import Path
from ultralytics import YOLO
import csv
import statistics


def main():
    ROOT = Path(r"F:\Program Files\Underwater_Detection")

    rows = []

    for i in range(1, 6):
        print("=" * 60)
        print(f"正在评估 YOLO11s_aug test_fold_{i} ...")

        model_path = ROOT / "runs/detect" / f"yolo11s_aug_train_fold_{i}" / "weights" / "best.pt"
        yaml_path = ROOT / "yolo_dataset_augmented" / f"fold_{i}.yaml"

        if not model_path.exists():
            print(f"缺失模型文件: {model_path}")
            continue

        if not yaml_path.exists():
            print(f"缺失 YAML 文件: {yaml_path}")
            continue

        model = YOLO(model_path)

        metrics = model.val(
            data=yaml_path.as_posix(),
            split="test",
            imgsz=640,
            batch=8,
            workers=0,
            device=0,
            project=(ROOT / "runs/detect").as_posix(),
            name=f"yolo11s_aug_test_fold_{i}_summary",
            exist_ok=True
        )

        row = {
            "fold": i,
            "Precision": metrics.box.mp,
            "Recall": metrics.box.mr,
            "mAP50": metrics.box.map50,
            "mAP50-95": metrics.box.map
        }

        rows.append(row)

    print("\nYOLO11s_aug 每折 test 结果：")
    print("fold\tPrecision\tRecall\tmAP50\tmAP50-95")

    for row in rows:
        print(
            f"{row['fold']}\t"
            f"{row['Precision']:.4f}\t\t"
            f"{row['Recall']:.4f}\t"
            f"{row['mAP50']:.4f}\t"
            f"{row['mAP50-95']:.4f}"
        )

    metrics_names = ["Precision", "Recall", "mAP50", "mAP50-95"]

    print("\nYOLO11s_aug test 五折平均值：")
    for name in metrics_names:
        values = [row[name] for row in rows]
        print(f"{name}: {statistics.mean(values):.4f}")

    print("\nYOLO11s_aug test 五折标准差：")
    for name in metrics_names:
        values = [row[name] for row in rows]
        if len(values) > 1:
            print(f"{name}: {statistics.stdev(values):.4f}")
        else:
            print(f"{name}: 0.0000")

    save_path = ROOT / "runs/detect/yolo11s_aug_kfold_test_summary.csv"

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["fold", "Precision", "Recall", "mAP50", "mAP50-95"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nYOLO11s_aug test 结果已保存到: {save_path}")


if __name__ == "__main__":
    main()