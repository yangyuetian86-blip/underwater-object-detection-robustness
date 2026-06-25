from pathlib import Path
from PIL import Image, ImageDraw
from ultralytics import YOLO


PROJECT_ROOT = Path(r"F:/Program Files/Underwater_Detection")

MODEL_CANDIDATES = [
    PROJECT_ROOT / "Archive_YOLO11s_5Fold_Results/yolo11s_train_fold_1/weights/best.pt",
    PROJECT_ROOT / "runs/detect/yolo11s_train_fold_1/weights/best.pt",
]

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

OUTPUT_ROOT = PROJECT_ROOT / "paper_figures/holothurian_missed_candidates"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "holothurian",
    1: "echinus",
    2: "scallop",
    3: "starfish",
}

HOLO_CLASS_ID = 0
CONF_THRES = 0.25
IOU_THRES = 0.5
IMG_SIZE = 640
DEVICE = 0
MAX_SAVE_PER_SET = 50


def find_model_path():
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path

    candidates = list(PROJECT_ROOT.rglob("yolo11s*fold_1*/weights/best.pt"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError("Cannot find YOLO11s fold 1 best.pt.")


def collect_images(image_dir):
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_paths.extend(image_dir.glob(ext))
    return sorted(image_paths)


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

        boxes.append({
            "cls": cls_id,
            "box": [x1, y1, x2, y2],
        })

    return boxes


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def draw_box(draw, box, text, color, width=3):
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

    text_y = max(0, y1 - 15)
    draw.rectangle(
        [x1, text_y, x1 + len(text) * 7 + 6, text_y + 15],
        fill=color,
    )
    draw.text((x1 + 2, text_y), text, fill="white")


def process_image(model, image_path, label_dir, save_path):
    image = Image.open(image_path).convert("RGB")
    image_w, image_h = image.size

    label_path = label_dir / f"{image_path.stem}.txt"
    gt_boxes = read_yolo_labels(label_path, image_w, image_h)

    gt_holo = [b for b in gt_boxes if b["cls"] == HOLO_CLASS_ID]

    if len(gt_holo) == 0:
        return False

    result = model.predict(
        source=image_path.as_posix(),
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        device=DEVICE,
        verbose=False,
    )[0]

    pred_boxes = []

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()

            pred_boxes.append({
                "cls": cls_id,
                "conf": conf,
                "box": xyxy,
            })

    pred_holo = [b for b in pred_boxes if b["cls"] == HOLO_CLASS_ID]

    missed_holo = []

    for gt in gt_holo:
        matched = False

        for pred in pred_holo:
            if iou(gt["box"], pred["box"]) >= IOU_THRES:
                matched = True
                break

        if not matched:
            missed_holo.append(gt)

    if len(missed_holo) == 0:
        return False

    draw = ImageDraw.Draw(image)

    # Draw all GT boxes in green
    for gt in gt_boxes:
        cls_id = gt["cls"]
        name = CLASS_NAMES.get(cls_id, str(cls_id))
        draw_box(draw, gt["box"], f"GT {name}", color="green", width=2)

    # Draw all predictions in red
    for pred in pred_boxes:
        cls_id = pred["cls"]
        name = CLASS_NAMES.get(cls_id, str(cls_id))
        conf = pred["conf"]
        draw_box(draw, pred["box"], f"Pred {name} {conf:.2f}", color="red", width=2)

    # Highlight missed holothurian in yellow
    for gt in missed_holo:
        draw_box(draw, gt["box"], "Missed holothurian", color="orange", width=4)

    image.save(save_path)

    return True


def main():
    model_path = find_model_path()
    print(f"Using model: {model_path}")

    model = YOLO(model_path.as_posix())

    for set_name, info in TEST_SETS.items():
        image_dir = info["image_dir"]
        label_dir = info["label_dir"]

        if not image_dir.exists():
            print(f"Skip {set_name}, image folder not found: {image_dir}")
            continue

        if not label_dir.exists():
            print(f"Skip {set_name}, label folder not found: {label_dir}")
            continue

        save_dir = OUTPUT_ROOT / set_name
        save_dir.mkdir(parents=True, exist_ok=True)

        image_paths = collect_images(image_dir)

        saved_count = 0

        print(f"Searching {set_name}...")

        for image_path in image_paths:
            save_path = save_dir / f"{image_path.stem}_missed_holothurian.jpg"

            is_saved = process_image(
                model=model,
                image_path=image_path,
                label_dir=label_dir,
                save_path=save_path,
            )

            if is_saved:
                saved_count += 1

            if saved_count >= MAX_SAVE_PER_SET:
                break

        print(f"{set_name}: saved {saved_count} missed holothurian candidates to {save_dir}")

    print("Finished.")


if __name__ == "__main__":
    main()