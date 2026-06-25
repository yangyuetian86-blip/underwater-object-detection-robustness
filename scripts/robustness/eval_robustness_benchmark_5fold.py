from pathlib import Path
import pandas as pd
from ultralytics import YOLO, RTDETR


PROJECT_ROOT = Path(r"F:/Program Files/Underwater_Detection")
DATASET_ROOT = PROJECT_ROOT / "yolo_dataset"
BENCHMARK_ROOT = DATASET_ROOT / "benchmark"

SAVE_ROOT = PROJECT_ROOT / "runs" / "benchmark_eval_5fold"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["holothurian", "echinus", "scallop", "starfish"]

BENCHMARK_SETS = [
    "blur_low",
    "blur_medium",
    "blur_high",
    "haze_low",
    "haze_medium",
    "haze_high",
    "blur_haze_low",
    "blur_haze_medium",
    "blur_haze_high",
]


MODEL_CONFIGS = [
    {
        "model_name": "YOLO11s_640",
        "model_type": "yolo",
        "imgsz": 640,
        "fold_paths": [
            PROJECT_ROOT / f"runs/detect/yolo11s_train_fold_{fold}/weights/best.pt"
            for fold in range(1, 6)
        ],
    },
    {
        "model_name": "YOLO11m_640",
        "model_type": "yolo",
        "imgsz": 640,
        "fold_paths": [
            PROJECT_ROOT / f"runs/detect/yolo11m_640_train_fold_{fold}/weights/best.pt"
            for fold in range(1, 6)
        ],
    },
    {
        "model_name": "RTDETR_l_640",
        "model_type": "rtdetr",
        "imgsz": 640,
        "fold_paths": [
            PROJECT_ROOT / f"runs/rtdetr/rtdetr_fold_{fold}/weights/best.pt"
            for fold in range(1, 6)
        ],
    },
]


def make_yaml(benchmark_name):
    image_dir = BENCHMARK_ROOT / benchmark_name / "images"
    label_dir = BENCHMARK_ROOT / benchmark_name / "labels"

    if not image_dir.exists():
        raise FileNotFoundError(f"找不到图片目录: {image_dir}")

    if not label_dir.exists():
        raise FileNotFoundError(f"找不到标签目录: {label_dir}")

    image_count = len(list(image_dir.glob("*.jpg")))
    label_count = len(list(label_dir.glob("*.txt")))

    print(f"{benchmark_name}: images={image_count}, labels={label_count}")

    if image_count == 0:
        raise RuntimeError(f"{benchmark_name} 图片数量为 0")

    if label_count == 0:
        raise RuntimeError(f"{benchmark_name} 标签数量为 0")

    yaml_path = SAVE_ROOT / f"{benchmark_name}.yaml"

    yaml_content = (
        f'train: "{(DATASET_ROOT / "images/train").absolute().as_posix()}"\n'
        f'val: "{image_dir.absolute().as_posix()}"\n'
        f'test: "{image_dir.absolute().as_posix()}"\n'
        f"\n"
        f"nc: 4\n"
        f"names: {CLASS_NAMES}\n"
    )

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    return yaml_path


def get_metrics(metrics):
    return {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
    }


def main():
    all_rows = []

    for benchmark_name in BENCHMARK_SETS:
        print("\n" + "=" * 100)
        print(f"Benchmark: {benchmark_name}")
        print("=" * 100)

        yaml_path = make_yaml(benchmark_name)

        for cfg in MODEL_CONFIGS:
            model_name = cfg["model_name"]
            model_type = cfg["model_type"]
            imgsz = cfg["imgsz"]
            fold_paths = cfg["fold_paths"]

            print("\n" + "-" * 100)
            print(f"Model: {model_name}")
            print("-" * 100)

            for fold_idx, model_path in enumerate(fold_paths, start=1):
                if not model_path.exists():
                    print(f"跳过 {model_name} fold {fold_idx}，找不到模型: {model_path}")
                    continue

                print(f"\n正在测试: {model_name} fold {fold_idx}")
                print(f"模型路径: {model_path}")

                if model_type == "yolo":
                    model = YOLO(model_path.as_posix())
                    batch = 8
                elif model_type == "rtdetr":
                    model = RTDETR(model_path.as_posix())
                    batch = 4
                else:
                    raise ValueError(f"未知模型类型: {model_type}")

                metrics = model.val(
                    data=yaml_path.as_posix(),
                    split="test",
                    imgsz=imgsz,
                    batch=batch,
                    workers=0,
                    device=0,
                    project=SAVE_ROOT.as_posix(),
                    name=f"{model_name}_{benchmark_name}_fold_{fold_idx}",
                    exist_ok=True,
                )

                row = {
                    "benchmark": benchmark_name,
                    "model": model_name,
                    "fold": fold_idx,
                    **get_metrics(metrics),
                }

                all_rows.append(row)
                print(row)

    df = pd.DataFrame(all_rows)

    raw_save_path = SAVE_ROOT / "robustness_results_5fold_raw.csv"
    df.to_csv(raw_save_path, index=False, encoding="utf-8-sig")

    summary_rows = []

    for (benchmark, model), group in df.groupby(["benchmark", "model"]):
        mean_row = {
            "benchmark": benchmark,
            "model": model,
            "stat": "mean",
            "precision": group["precision"].mean(),
            "recall": group["recall"].mean(),
            "mAP50": group["mAP50"].mean(),
            "mAP50-95": group["mAP50-95"].mean(),
        }

        std_row = {
            "benchmark": benchmark,
            "model": model,
            "stat": "std",
            "precision": group["precision"].std(),
            "recall": group["recall"].std(),
            "mAP50": group["mAP50"].std(),
            "mAP50-95": group["mAP50-95"].std(),
        }

        summary_rows.append(mean_row)
        summary_rows.append(std_row)

    summary_df = pd.DataFrame(summary_rows)

    summary_save_path = SAVE_ROOT / "robustness_results_5fold_summary.csv"
    summary_df.to_csv(summary_save_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("五折鲁棒性测试完成")
    print(f"原始 fold 结果保存到: {raw_save_path}")
    print(f"mean/std 汇总结果保存到: {summary_save_path}")
    print("=" * 100)

    print(summary_df)


if __name__ == "__main__":
    main()