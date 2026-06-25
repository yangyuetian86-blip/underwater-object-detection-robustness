from pathlib import Path

ROOT = Path(r"F:\Program Files\Underwater_Detection")

# 你可以改成 test 来检查测试集
IMG_DIR = ROOT / "yolo_dataset/images/train"
LABEL_DIR = ROOT / "yolo_dataset/labels/train"

#IMG_DIR = ROOT / "yolo_dataset/images/test"
#LABEL_DIR = ROOT / "yolo_dataset/labels/test"

CLASS_NAMES = {
    0: "holothurian",
    1: "echinus",
    2: "scallop",
    3: "starfish",
}

small_box_threshold = 0.0015  # 框面积过小阈值，w*h


def main():
    image_files = sorted(IMG_DIR.glob("*.jpg"))
    label_files = sorted(LABEL_DIR.glob("*.txt"))

    image_stems = {p.stem for p in image_files}
    label_stems = {p.stem for p in label_files}

    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)

    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    tiny_class_counts = {0: 0, 1: 0, 2: 0, 3: 0}


    empty_labels = []
    invalid_lines = []
    out_of_range = []
    tiny_boxes = []

    total_boxes = 0

    for label_path in label_files:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if len(lines) == 0:
            empty_labels.append(label_path.name)
            continue

        for line_idx, line in enumerate(lines, start=1):
            parts = line.split()

            if len(parts) != 5:
                invalid_lines.append((label_path.name, line_idx, line))
                continue

            try:
                cls_id = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
            except Exception:
                invalid_lines.append((label_path.name, line_idx, line))
                continue

            total_boxes += 1

            if cls_id in class_counts:
                class_counts[cls_id] += 1
            else:
                invalid_lines.append((label_path.name, line_idx, f"invalid class: {cls_id}"))

            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                out_of_range.append((label_path.name, line_idx, line))

            if w * h < small_box_threshold:
                tiny_boxes.append((label_path.name, line_idx, line, w * h))
                if cls_id in tiny_class_counts:
                    tiny_class_counts[cls_id] += 1
    print("=" * 70)
    print("标签质量自动检查结果")
    print("=" * 70)

    print(f"图片数量: {len(image_files)}")
    print(f"标签数量: {len(label_files)}")
    print(f"总框数: {total_boxes}")

    print("\n类别框数量:")
    for cls_id, count in class_counts.items():
        print(f"{cls_id} {CLASS_NAMES[cls_id]}: {count}")

    print("\n极小框数量:")
    for cls_id, count in tiny_class_counts.items():
        print(f"{cls_id} {CLASS_NAMES[cls_id]}: {count}")

    print("\n异常统计:")
    print(f"缺失标签图片数量: {len(missing_labels)}")
    print(f"缺失图片标签数量: {len(missing_images)}")
    print(f"空标签文件数量: {len(empty_labels)}")
    print(f"格式错误行数量: {len(invalid_lines)}")
    print(f"坐标越界框数量: {len(out_of_range)}")
    print(f"极小框数量: {len(tiny_boxes)}")

    print("\n前 20 个极小框:")
    for item in tiny_boxes[:20]:
        print(item)

    print("\n前 20 个坐标越界框:")
    for item in out_of_range[:20]:
        print(item)

    print("\n前 20 个格式错误:")
    for item in invalid_lines[:20]:
        print(item)


if __name__ == "__main__":
    main()