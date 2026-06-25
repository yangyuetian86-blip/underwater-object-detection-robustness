from pathlib import Path
import csv
import statistics


def main():
    ROOT = Path(r"F:\Program Files\Underwater_Detection")
    runs_dir = ROOT / "runs/detect"

    rows = []

    for i in range(1, 6):
        csv_path = runs_dir / f"yolo11s_aug_train_fold_{i}" / "results.csv"

        if not csv_path.exists():
            print(f"缺失文件: {csv_path}")
            continue

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = list(reader)

        if not data:
            print(f"空文件: {csv_path}")
            continue

        # 取 mAP50-95 最高的 epoch，和 best.pt 对应更接近
        best = max(
            data,
            key=lambda r: float(r["metrics/mAP50-95(B)"])
        )

        row = {
            "fold": i,
            "best_epoch": best["epoch"],
            "Precision": float(best["metrics/precision(B)"]),
            "Recall": float(best["metrics/recall(B)"]),
            "mAP50": float(best["metrics/mAP50(B)"]),
            "mAP50-95": float(best["metrics/mAP50-95(B)"])
        }

        rows.append(row)

    print("\nYOLO11s_aug 每折 val 结果：")
    print("fold\tbest_epoch\tPrecision\tRecall\tmAP50\tmAP50-95")

    for row in rows:
        print(
            f"{row['fold']}\t"
            f"{row['best_epoch']}\t\t"
            f"{row['Precision']:.4f}\t\t"
            f"{row['Recall']:.4f}\t"
            f"{row['mAP50']:.4f}\t"
            f"{row['mAP50-95']:.4f}"
        )

    metrics_names = ["Precision", "Recall", "mAP50", "mAP50-95"]

    print("\nYOLO11s_aug val 五折平均值：")
    for name in metrics_names:
        values = [row[name] for row in rows]
        print(f"{name}: {statistics.mean(values):.4f}")

    print("\nYOLO11s_aug val 五折标准差：")
    for name in metrics_names:
        values = [row[name] for row in rows]
        if len(values) > 1:
            print(f"{name}: {statistics.stdev(values):.4f}")
        else:
            print(f"{name}: 0.0000")

    save_path = runs_dir / "yolo11s_aug_kfold_val_summary.csv"

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["fold", "best_epoch", "Precision", "Recall", "mAP50", "mAP50-95"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nYOLO11s_aug val 结果已保存到: {save_path}")


if __name__ == "__main__":
    main()