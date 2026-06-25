from pathlib import Path
import cv2
import random

ROOT = Path(r"F:\Program Files\Underwater_Detection")

IMG_DIR = ROOT / "yolo_dataset/images/train"
LABEL_DIR = ROOT / "yolo_dataset/labels/train"

OUT_DIR = ROOT / "label_visual_check_holothurian_scallop"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "holothurian",
    1: "echinus",
    2: "scallop",
    3: "starfish",
}

TARGET_CLASSES = {0, 2}


def yolo_to_xyxy(x, y, w, h, img_w, img_h):
    x1 = int((x - w / 2) * img_w)
    y1 = int((y - h / 2) * img_h)
    x2 = int((x + w / 2) * img_w)
    y2 = int((y + h / 2) * img_h)

    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    return x1, y1, x2, y2


def has_target(label_path):
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue

        cls_id = int(float(parts[0]))
        if cls_id in TARGET_CLASSES:
            return True

    return False


def draw_image(img_path, label_path, out_path):
    img = cv2.imread(str(img_path))

    if img is None:
        print(f"无法读取图片: {img_path}")
        return False

    img_h, img_w = img.shape[:2]

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue

        cls_id = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:])

        x1, y1, x2, y2 = yolo_to_xyxy(x, y, w, h, img_w, img_h)

        if cls_id == 0:
            color = (0, 255, 255)      # holothurian 黄色
        elif cls_id == 2:
            color = (255, 0, 255)      # scallop 紫色
        else:
            color = (0, 255, 0)        # 其他类别绿色

        thickness = 3 if cls_id in TARGET_CLASSES else 1
        text = f"{cls_id}:{CLASS_NAMES.get(cls_id, 'unknown')}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            img,
            text,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )

    cv2.imwrite(str(out_path), img)
    return True


def main():
    label_files = sorted(LABEL_DIR.glob("*.txt"))

    target_files = [
        label_path for label_path in label_files
        if has_target(label_path)
    ]

    print(f"包含 holothurian/scallop 的图片数量: {len(target_files)}")

    sample_num = min(200, len(target_files))
    sampled_files = random.sample(target_files, sample_num)

    saved = 0

    for label_path in sampled_files:
        img_path = IMG_DIR / f"{label_path.stem}.jpg"
        out_path = OUT_DIR / f"{label_path.stem}_check.jpg"

        if not img_path.exists():
            print(f"缺失图片: {img_path}")
            continue

        if draw_image(img_path, label_path, out_path):
            saved += 1

    print(f"已生成可视化检查图片: {saved}")
    print(f"输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()