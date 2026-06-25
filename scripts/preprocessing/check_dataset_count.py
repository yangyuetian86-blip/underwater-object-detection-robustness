import os
from pathlib import Path

ROOT = Path(r"F:\Program Files\Underwater_Detection")

original_img_dir = ROOT / "original_data" / "image"
original_box_dir = ROOT / "original_data" / "box"

train_img_dir = ROOT / "yolo_dataset" / "images" / "train"
train_lab_dir = ROOT / "yolo_dataset" / "labels" / "train"

test_img_dir = ROOT / "yolo_dataset" / "images" / "test"
test_lab_dir = ROOT / "yolo_dataset" / "labels" / "test"


def get_stems(folder, exts):
    stems = set()
    files = []
    for ext in exts:
        files.extend(folder.glob(ext))
    for f in files:
        stems.add(f.stem)
    return stems


img_exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP"]

original_imgs = get_stems(original_img_dir, img_exts)
train_imgs = get_stems(train_img_dir, img_exts)
test_imgs = get_stems(test_img_dir, img_exts)

train_labs = get_stems(train_lab_dir, ["*.txt"])
test_labs = get_stems(test_lab_dir, ["*.txt"])

yolo_imgs = train_imgs | test_imgs

print("=" * 70)
print(f"original_data/image 图片数: {len(original_imgs)}")
print(f"yolo train 图片数:          {len(train_imgs)}")
print(f"yolo test 图片数:           {len(test_imgs)}")
print(f"yolo train+test 图片数:     {len(yolo_imgs)}")
print("=" * 70)

missing_from_yolo = sorted(list(original_imgs - yolo_imgs))
extra_in_yolo = sorted(list(yolo_imgs - original_imgs))

print(f"原始图片中有，但 yolo_dataset 里没有的数量: {len(missing_from_yolo)}")
print(f"yolo_dataset 中有，但 original_data 里没有的数量: {len(extra_in_yolo)}")

if missing_from_yolo:
    print("\n前 50 个缺失文件名：")
    for name in missing_from_yolo[:50]:
        print(name)

print("=" * 70)

train_missing_labels = sorted(list(train_imgs - train_labs))
test_missing_labels = sorted(list(test_imgs - test_labs))

print(f"train 图片缺失标签数量: {len(train_missing_labels)}")
print(f"test 图片缺失标签数量:  {len(test_missing_labels)}")

if train_missing_labels:
    print("\ntrain 缺失标签前 30 个：")
    for name in train_missing_labels[:30]:
        print(name)

if test_missing_labels:
    print("\ntest 缺失标签前 30 个：")
    for name in test_missing_labels[:30]:
        print(name)

print("=" * 70)

overlap = train_imgs & test_imgs
print(f"train/test 重名重叠数量: {len(overlap)}")

if overlap:
    print("重叠文件前 30 个：")
    for name in list(overlap)[:30]:
        print(name)