from pathlib import Path
import torch
import pandas as pd
from PIL import Image

import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader

from torchmetrics.detection.mean_ap import MeanAveragePrecision


PROJECT_ROOT = Path(r"F:/Program Files/Underwater_Detection")
DATASET_ROOT = PROJECT_ROOT / "yolo_dataset"
BENCHMARK_ROOT = DATASET_ROOT / "benchmark"

MODEL_PATH = PROJECT_ROOT / "runs/faster_rcnn/faster_rcnn_fold_1/best_model.pth"

SAVE_ROOT = PROJECT_ROOT / "runs" / "faster_rcnn_robustness_eval"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 5
# background = 0
# holothurian = 1
# echinus = 2
# scallop = 3
# starfish = 4

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


class YOLODetectionDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

        self.image_paths = sorted(list(self.image_dir.glob("*.jpg")))

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_dir / f"{image_path.stem}.txt"

        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                cls_id = int(float(parts[0]))
                x_center = float(parts[1]) * w
                y_center = float(parts[2]) * h
                box_w = float(parts[3]) * w
                box_h = float(parts[4]) * h

                x1 = x_center - box_w / 2
                y1 = y_center - box_h / 2
                x2 = x_center + box_w / 2
                y2 = y_center + box_h / 2

                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append([x1, y1, x2, y2])

                # torchvision detection label starts from 1
                labels.append(cls_id + 1)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        image_tensor = F.to_tensor(image)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_faster_rcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes
    )

    return model


def load_model(device):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到 Faster R-CNN 模型: {MODEL_PATH}")

    model = build_faster_rcnn_model(NUM_CLASSES)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(f"Loaded model from: {MODEL_PATH}")

    return model


@torch.no_grad()
def evaluate_one_benchmark(model, benchmark_name, device):
    image_dir = BENCHMARK_ROOT / benchmark_name / "images"
    label_dir = BENCHMARK_ROOT / benchmark_name / "labels"

    if not image_dir.exists():
        raise FileNotFoundError(f"找不到图片目录: {image_dir}")

    if not label_dir.exists():
        raise FileNotFoundError(f"找不到标签目录: {label_dir}")

    image_count = len(list(image_dir.glob("*.jpg")))
    label_count = len(list(label_dir.glob("*.txt")))

    print("\n" + "=" * 80)
    print(f"Benchmark: {benchmark_name}")
    print(f"Images: {image_count}, Labels: {label_count}")
    print("=" * 80)

    dataset = YOLODetectionDataset(
        image_dir=image_dir,
        label_dir=label_dir
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=False
    )

    for images, targets in dataloader:
        images = [img.to(device) for img in images]

        outputs = model(images)

        preds = []
        gts = []

        for output, target in zip(outputs, targets):
            preds.append({
                "boxes": output["boxes"].detach().cpu(),
                "scores": output["scores"].detach().cpu(),
                "labels": output["labels"].detach().cpu(),
            })

            gts.append({
                "boxes": target["boxes"].detach().cpu(),
                "labels": target["labels"].detach().cpu(),
            })

        metric.update(preds, gts)

    result = metric.compute()

    row = {
        "benchmark": benchmark_name,
        "model": "Faster_R-CNN_ResNet50_FPN_fold1",
        "precision": float(result["map"]),  # torchmetrics 不直接给 YOLO 式 precision
        "recall": float(result["mar_100"]),
        "mAP50": float(result["map_50"]),
        "mAP50-95": float(result["map"]),
    }

    print(row)

    return row


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    model = load_model(device)

    rows = []

    for benchmark_name in BENCHMARK_SETS:
        row = evaluate_one_benchmark(
            model=model,
            benchmark_name=benchmark_name,
            device=device
        )

        rows.append(row)

    df = pd.DataFrame(rows)

    save_path = SAVE_ROOT / "faster_rcnn_robustness_results_fold1.csv"
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("Faster R-CNN robustness evaluation finished.")
    print(f"Saved to: {save_path}")
    print("=" * 80)
    print(df)


if __name__ == "__main__":
    main()