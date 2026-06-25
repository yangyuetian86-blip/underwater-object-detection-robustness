import time
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchmetrics.detection.mean_ap import MeanAveragePrecision


# =========================================================
# 1. Basic config
# =========================================================

PROJECT_ROOT = Path(r"F:/Program Files/Underwater_Detection")
DATASET_ROOT = PROJECT_ROOT / "yolo_dataset"

TRAIN_IMAGES_DIR = DATASET_ROOT / "images/train"
TRAIN_LABELS_DIR = DATASET_ROOT / "labels/train"

TEST_IMAGES_DIR = DATASET_ROOT / "images/test"
TEST_LABELS_DIR = DATASET_ROOT / "labels/test"

RUNS_DIR = PROJECT_ROOT / "runs/faster_rcnn"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    1: "holothurian",
    2: "echinus",
    3: "scallop",
    4: "starfish",
}

NUM_CLASSES = 5  # 4 classes + background

K = 5
SEED = 42

# Faster R-CNN 很慢，作为 classic CNN baseline，建议只跑 fold 1
FOLDS_TO_RUN = [1]

EPOCHS = 30
VAL_INTERVAL = 5

BATCH_SIZE = 4
NUM_WORKERS = 2

LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

CONF_THRES = 0.25
IOU_THRES = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 2. YOLO label to xyxy
# =========================================================

def yolo_to_xyxy(label_line, img_w, img_h):
    parts = label_line.strip().split()

    if len(parts) != 5:
        raise ValueError(f"Invalid YOLO label line: {label_line}")

    cls = int(float(parts[0]))

    x_center = float(parts[1]) * img_w
    y_center = float(parts[2]) * img_h
    box_w = float(parts[3]) * img_w
    box_h = float(parts[4]) * img_h

    x1 = x_center - box_w / 2
    y1 = y_center - box_h / 2
    x2 = x_center + box_w / 2
    y2 = y_center + box_h / 2

    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    # torchvision detection labels must start from 1
    label = cls + 1

    return [x1, y1, x2, y2], label


def collate_fn(batch):
    return tuple(zip(*batch))


# =========================================================
# 3. Dataset
# =========================================================

class UnderwaterYoloDetectionDataset(Dataset):
    def __init__(self, image_dir, label_dir, file_stems):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.file_stems = list(file_stems)

    def __len__(self):
        return len(self.file_stems)

    def __getitem__(self, idx):
        file_stem = self.file_stems[idx]

        img_path = self.image_dir / f"{file_stem}.jpg"
        label_path = self.label_dir / f"{file_stem}.txt"

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        boxes = []
        labels = []

        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            if line.strip() == "":
                continue

            box, label = yolo_to_xyxy(line, img_w, img_h)
            x1, y1, x2, y2 = box

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append(box)
            labels.append(label)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
        }

        img = F.to_tensor(img)

        return img, target


# =========================================================
# 4. Model
# =========================================================

def get_faster_rcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT"
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes
    )

    return model


# =========================================================
# 5. Precision / Recall calculation
# =========================================================

def box_iou_single(box1, box2):
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
    area2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))

    union = area1 + area2 - inter

    if union <= 0:
        return 0.0

    return inter / union


def compute_precision_recall(preds, targets, conf_thres=0.25, iou_thres=0.5):
    tp = 0
    fp = 0
    fn = 0

    per_class_stats = {
        cls_id: {"tp": 0, "fp": 0, "fn": 0}
        for cls_id in CLASS_NAMES.keys()
    }

    for pred, target in zip(preds, targets):
        pred_boxes = pred["boxes"].detach().cpu()
        pred_scores = pred["scores"].detach().cpu()
        pred_labels = pred["labels"].detach().cpu()

        gt_boxes = target["boxes"].detach().cpu()
        gt_labels = target["labels"].detach().cpu()

        keep = pred_scores >= conf_thres

        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]

        matched_gt = set()
        order = torch.argsort(pred_scores, descending=True)

        for pred_idx in order:
            p_box = pred_boxes[pred_idx]
            p_label = int(pred_labels[pred_idx])

            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue

                if int(gt_labels[gt_idx]) != p_label:
                    continue

                iou = box_iou_single(p_box, gt_box)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_thres and best_gt_idx >= 0:
                tp += 1
                matched_gt.add(best_gt_idx)

                if p_label in per_class_stats:
                    per_class_stats[p_label]["tp"] += 1
            else:
                fp += 1

                if p_label in per_class_stats:
                    per_class_stats[p_label]["fp"] += 1

        for gt_idx, gt_label in enumerate(gt_labels):
            if gt_idx not in matched_gt:
                fn += 1

                gt_label = int(gt_label)

                if gt_label in per_class_stats:
                    per_class_stats[gt_label]["fn"] += 1

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)

    per_class_result = {}

    for cls_id, stats in per_class_stats.items():
        cls_tp = stats["tp"]
        cls_fp = stats["fp"]
        cls_fn = stats["fn"]

        cls_p = cls_tp / (cls_tp + cls_fp + 1e-9)
        cls_r = cls_tp / (cls_tp + cls_fn + 1e-9)

        per_class_result[cls_id] = {
            "precision": cls_p,
            "recall": cls_r,
        }

    return precision, recall, per_class_result


# =========================================================
# 6. Train / Eval functions
# =========================================================

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()

    total_loss = 0.0
    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]

        targets = [
            {k: v.to(device) for k, v in t.items()}
            for t in targets
        ]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += float(losses.item())

        if (batch_idx + 1) % 50 == 0:
            print(
                f"Epoch [{epoch}] "
                f"Batch [{batch_idx + 1}/{len(data_loader)}] "
                f"Loss: {losses.item():.4f}"
            )

    avg_loss = total_loss / max(1, len(data_loader))
    elapsed = time.time() - start_time

    print(
        f"Epoch [{epoch}] finished. "
        f"Avg loss: {avg_loss:.4f}. "
        f"Time: {elapsed:.1f}s"
    )

    return avg_loss


@torch.no_grad()
def evaluate_model(model, data_loader, device):
    model.eval()

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True
    )

    all_preds = []
    all_targets = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]

        outputs = model(images)

        preds_cpu = []
        targets_cpu = []

        for output, target in zip(outputs, targets):
            pred = {
                "boxes": output["boxes"].detach().cpu(),
                "scores": output["scores"].detach().cpu(),
                "labels": output["labels"].detach().cpu(),
            }

            tgt = {
                "boxes": target["boxes"].detach().cpu(),
                "labels": target["labels"].detach().cpu(),
            }

            preds_cpu.append(pred)
            targets_cpu.append(tgt)

        metric.update(preds_cpu, targets_cpu)
        all_preds.extend(preds_cpu)
        all_targets.extend(targets_cpu)

    map_result = metric.compute()

    precision, recall, per_class_pr = compute_precision_recall(
        all_preds,
        all_targets,
        conf_thres=CONF_THRES,
        iou_thres=IOU_THRES
    )

    results = {
        "precision": precision,
        "recall": recall,
        "mAP50": float(map_result["map_50"]),
        "mAP50-95": float(map_result["map"]),
        "per_class_pr": per_class_pr,
        "map_result": map_result,
    }

    return results


def print_eval_results(results, title="Evaluation"):
    """
    Safe print version.
    It only prints overall mAP50 and mAP50-95.
    For each class, it prints Precision / Recall only,
    because torchmetrics may not always return map_50_per_class.
    """
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(
        f"all | "
        f"P: {results['precision']:.4f} | "
        f"R: {results['recall']:.4f} | "
        f"mAP50: {results['mAP50']:.4f} | "
        f"mAP50-95: {results['mAP50-95']:.4f}"
    )

    print("\nPer-class Precision / Recall:")

    for cls_id, cls_name in CLASS_NAMES.items():
        cls_pr = results["per_class_pr"].get(
            cls_id,
            {"precision": 0.0, "recall": 0.0}
        )

        print(
            f"{cls_name} | "
            f"P: {cls_pr['precision']:.4f} | "
            f"R: {cls_pr['recall']:.4f}"
        )


# =========================================================
# 7. Main
# =========================================================

def main():
    print("=" * 70)
    print("Faster R-CNN ResNet50-FPN baseline")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    all_files = sorted([
        f.stem for f in TRAIN_IMAGES_DIR.glob("*.jpg")
    ])

    test_files = sorted([
        f.stem for f in TEST_IMAGES_DIR.glob("*.jpg")
    ])

    print(f"Trainval images: {len(all_files)}")
    print(f"Test images    : {len(test_files)}")

    if len(all_files) == 0:
        raise RuntimeError("No training images found.")

    if len(test_files) == 0:
        raise RuntimeError("No test images found.")

    overlap = set(all_files) & set(test_files)

    if len(overlap) > 0:
        raise RuntimeError("Train/test overlap detected.")

    np.random.seed(SEED)
    np.random.shuffle(all_files)

    folds = np.array_split(all_files, K)

    test_summary_rows = []

    for fold in range(1, K + 1):
        if fold not in FOLDS_TO_RUN:
            continue

        print("\n" + "=" * 70)
        print(f"Starting Faster R-CNN fold {fold}/{K}")
        print("=" * 70)

        val_files = list(folds[fold - 1])

        train_files = []

        for j in range(K):
            if j != fold - 1:
                train_files.extend(folds[j])

        print(f"Train images: {len(train_files)}")
        print(f"Val images  : {len(val_files)}")
        print(f"Test images : {len(test_files)}")

        train_dataset = UnderwaterYoloDetectionDataset(
            TRAIN_IMAGES_DIR,
            TRAIN_LABELS_DIR,
            train_files
        )

        val_dataset = UnderwaterYoloDetectionDataset(
            TRAIN_IMAGES_DIR,
            TRAIN_LABELS_DIR,
            val_files
        )

        test_dataset = UnderwaterYoloDetectionDataset(
            TEST_IMAGES_DIR,
            TEST_LABELS_DIR,
            test_files
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn
        )

        model = get_faster_rcnn_model(NUM_CLASSES)
        model.to(DEVICE)

        params = [
            p for p in model.parameters()
            if p.requires_grad
        ]

        optimizer = torch.optim.SGD(
            params,
            lr=LR,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY
        )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=5,
            gamma=0.1
        )

        fold_dir = RUNS_DIR / f"faster_rcnn_fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        best_map = -1.0
        best_path = fold_dir / "best_model.pth"

        for epoch in range(1, EPOCHS + 1):
            train_one_epoch(
                model,
                optimizer,
                train_loader,
                DEVICE,
                epoch
            )

            lr_scheduler.step()

            if epoch % VAL_INTERVAL == 0 or epoch == EPOCHS:
                val_results = evaluate_model(
                    model,
                    val_loader,
                    DEVICE
                )

                print_eval_results(
                    val_results,
                    title=f"Fold {fold} Val Epoch {epoch}"
                )

                if val_results["mAP50-95"] > best_map:
                    best_map = val_results["mAP50-95"]

                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "epoch": epoch,
                            "best_map": best_map,
                        },
                        best_path
                    )

                    print(f"Best model saved: {best_path}")

        print(f"\nLoading best model for fold {fold} test...")

        checkpoint = torch.load(
            best_path,
            map_location=DEVICE
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        test_results = evaluate_model(
            model,
            test_loader,
            DEVICE
        )

        print_eval_results(
            test_results,
            title=f"Fold {fold} Test"
        )

        test_row = {
            "fold": fold,
            "precision": test_results["precision"],
            "recall": test_results["recall"],
            "mAP50": test_results["mAP50"],
            "mAP50-95": test_results["mAP50-95"],
        }

        test_summary_rows.append(test_row)

        fold_result_path = fold_dir / "test_result.csv"

        pd.DataFrame([test_row]).to_csv(
            fold_result_path,
            index=False,
            encoding="utf-8-sig"
        )

        print(f"Fold test result saved: {fold_result_path}")

    if len(test_summary_rows) > 0:
        df = pd.DataFrame(test_summary_rows)

        if len(df) > 1:
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

        summary_path = RUNS_DIR / "faster_rcnn_test_summary.csv"

        df.to_csv(
            summary_path,
            index=False,
            encoding="utf-8-sig"
        )

        print("\n" + "=" * 70)
        print(f"Summary saved: {summary_path}")
        print(df)

    print("\nFaster R-CNN experiment finished.")


if __name__ == "__main__":
    main()