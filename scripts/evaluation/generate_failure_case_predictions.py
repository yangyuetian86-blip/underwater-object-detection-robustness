from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


PROJECT_ROOT = Path(r"F:/Program Files/Underwater_Detection")

# 优先使用 YOLO11s fold 1 作为 baseline
MODEL_CANDIDATES = [
    PROJECT_ROOT / "Archive_YOLO11s_5Fold_Results/yolo11s_train_fold_1/weights/best.pt",
    PROJECT_ROOT / "runs/detect/yolo11s_train_fold_1/weights/best.pt",
]

OUTPUT_ROOT = PROJECT_ROOT / "paper_figures/failure_case_candidates"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "holothurian",
    1: "echinus",
    2: "scallop",
    3: "starfish",
}

TEST_SETS = {
    "clean": {
        "image_dir": PROJECT_ROOT / "yolo_dataset/images/test",
        "label_dir": PROJECT_ROOT / "yolo_dataset/labels/test",
    },
    "haze_high": {
        "image_dir": PROJECT_ROOT / "yolo_dataset/benchmark/haze_high/images",
        "label_dir": PROJECT_ROOT / "yolo_dataset/benchmark/haze_high/labels",
    },
    "blur_haze_high": {
        "image_dir": PROJECT_ROOT / "yolo_dataset/benchmark/blur_haze_high/images",
        "label_dir": PROJECT_ROOT / "yolo_dataset/benchmark/blur_haze_high/labels",
    },
}

SAMPLE_NUM = 60
RANDOM_SEED = 42
CONF_THRES = 0.25
IMG_SIZE = 640
DEVICE = 0


def find_model_path():
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path

    candidates = list(PROJECT_ROOT.rglob("yolo11s*fold_1*/weights/best.pt"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError("Cannot find YOLO11s fold 1 best.pt. Please set MODEL_PATH manually.")


def read_yolo_labels(label_path, image_w, image_h):
    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        cls_id = int(float(parts[0]))
        x_center = float(parts[1]) * image_w
        y_center = float(parts[2]) * image_h
        box_w = float(parts[3]) * image_w
        box_h = float(parts[4]) * image_h

        x1 = x_center - box_w / 2
        y1 = y_center - box_h / 2
        x2 = x_center + box_w / 2
        y2 = y_center + box_h / 2

        boxes.append((cls_id, x1, y1, x2, y2))

    return boxes


def draw_box(draw, box, text, color, width=3):
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

    text_x = x1
    text_y = max(0, y1 - 14)

    draw.rectangle(
        [text_x, text_y, text_x + len(text) * 7 + 4, text_y + 14],
        fill=color,
    )
    draw.text((text_x + 2, text_y), text, fill="white")


def draw_gt_and_predictions(model, image_path, label_dir, save_path):
    image = Image.open(image_path).convert("RGB")
    image_w, image_h = image.size
    draw = ImageDraw.Draw(image)

    label_path = label_dir / f"{image_path.stem}.txt"

    gt_boxes = read_yolo_labels(label_path, image_w, image_h)

    # Green boxes: ground truth
    for cls_id, x1, y1, x2, y2 in gt_boxes:
        name = CLASS_NAMES.get(cls_id, str(cls_id))
        draw_box(
            draw,
            (x1, y1, x2, y2),
            f"GT {name}",
            color="green",
            width=3,
        )

    result = model.predict(
        source=image_path.as_posix(),
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        device=DEVICE,
        verbose=False,
    )[0]

    # Red boxes: predictions
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            name = CLASS_NAMES.get(cls_id, str(cls_id))
            draw_box(
                draw,
                (x1, y1, x2, y2),
                f"Pred {name} {conf:.2f}",
                color="red",
                width=3,
            )

    image.save(save_path)


def collect_images(image_dir):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_paths = []

    for ext in exts:
        image_paths.extend(image_dir.glob(ext))

    return sorted(image_paths)


def main():
    model_path = find_model_path()
    print(f"Using model: {model_path}")

    model = YOLO(model_path.as_posix())

    random.seed(RANDOM_SEED)

    for set_name, info in TEST_SETS.items():
        image_dir = info["image_dir"]
        label_dir = info["label_dir"]

        if not image_dir.exists():
            print(f"Skip {set_name}, image folder not found: {image_dir}")
            continue

        if not label_dir.exists():
            print(f"Skip {set_name}, label folder not found: {label_dir}")
            continue

        image_paths = collect_images(image_dir)

        if len(image_paths) == 0:
            print(f"Skip {set_name}, no images found.")
            continue

        selected = random.sample(image_paths, min(SAMPLE_NUM, len(image_paths)))

        save_dir = OUTPUT_ROOT / set_name
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {set_name}: {len(selected)} images")

        for i, image_path in enumerate(selected, start=1):
            save_path = save_dir / f"{image_path.stem}_gt_pred.jpg"
            draw_gt_and_predictions(model, image_path, label_dir, save_path)

            if i % 10 == 0:
                print(f"  {i}/{len(selected)} done")

        print(f"Saved to: {save_dir}")

    print("Finished.")


if __name__ == "__main__":
    main()