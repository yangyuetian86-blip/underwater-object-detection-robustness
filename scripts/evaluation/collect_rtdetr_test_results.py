from pathlib import Path
import pandas as pd
from ultralytics import RTDETR


PROJECT_ROOT = Path(r"F:/Program Files/Underwater_Detection")
DATASET_ROOT = PROJECT_ROOT / "yolo_dataset"
RUNS_DIR = PROJECT_ROOT / "runs/rtdetr"

rows = []

for fold in range(1, 6):
    model_path = RUNS_DIR / f"rtdetr_fold_{fold}" / "weights" / "best.pt"
    yaml_path = DATASET_ROOT / f"fold_{fold}.yaml"

    if not model_path.exists():
        print(f"⚠️ 找不到模型: {model_path}")
        continue

    if not yaml_path.exists():
        print(f"⚠️ 找不到 YAML: {yaml_path}")
        continue

    print(f"\n正在评估 RT-DETR fold {fold}...")

    model = RTDETR(model_path.as_posix())

    metrics = model.val(
        data=yaml_path.absolute().as_posix(),
        split="test",
        imgsz=640,
        batch=4,
        workers=0,
        device=0,
        project=RUNS_DIR.as_posix(),
        name=f"rtdetr_test_fold_{fold}_summary_check",
        exist_ok=True
    )

    row = {
        "fold": fold,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
    }

    rows.append(row)

df = pd.DataFrame(rows)

mean_row = {
    "fold": "mean",
    "precision": df["precision"].mean(),
    "recall": df["recall"].mean(),
    "mAP50": df["mAP50"].mean(),
    "mAP50-95": df["mAP50-95"].mean(),
}

std_row = {
    "fold": "std",
    "precision": df["precision"].std(),
    "recall": df["recall"].std(),
    "mAP50": df["mAP50"].std(),
    "mAP50-95": df["mAP50-95"].std(),
}

df = pd.concat(
    [df, pd.DataFrame([mean_row, std_row])],
    ignore_index=True
)

save_path = RUNS_DIR / "rtdetr_kfold_test_summary.csv"
df.to_csv(save_path, index=False, encoding="utf-8-sig")

print(f"\n✅ RT-DETR test summary 已保存: {save_path}")
print(df)