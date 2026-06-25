import os
import random
import shutil


# =========================
# 配置路径
# =========================

src_img_dir = r'F:\Program Files\Underwater_Detection\yolo_dataset\images\train'
src_lab_dir = r'F:\Program Files\Underwater_Detection\yolo_dataset\labels\train'

test_img_dir = r'F:\Program Files\Underwater_Detection\yolo_dataset\images\test'
test_lab_dir = r'F:\Program Files\Underwater_Detection\yolo_dataset\labels\test'

test_count = 550
random_seed = 42


# =========================
# 第一步：把旧 test 集归还到 train
# =========================

os.makedirs(src_img_dir, exist_ok=True)
os.makedirs(src_lab_dir, exist_ok=True)
os.makedirs(test_img_dir, exist_ok=True)
os.makedirs(test_lab_dir, exist_ok=True)

old_test_imgs = [f for f in os.listdir(test_img_dir) if f.lower().endswith(".jpg")]

if len(old_test_imgs) > 0:
    print(f"检测到旧 test 集：{len(old_test_imgs)} 张，正在归还到 train...")

    for img in old_test_imgs:
        old_img_path = os.path.join(test_img_dir, img)
        new_img_path = os.path.join(src_img_dir, img)

        if os.path.exists(old_img_path):
            shutil.move(old_img_path, new_img_path)

        lab = os.path.splitext(img)[0] + ".txt"
        old_lab_path = os.path.join(test_lab_dir, lab)
        new_lab_path = os.path.join(src_lab_dir, lab)

        if os.path.exists(old_lab_path):
            shutil.move(old_lab_path, new_lab_path)
        else:
            print(f"⚠️ 旧 test 图片缺失标签: {lab}")

    print(f"✅ 已将旧 test 集 {len(old_test_imgs)} 张图片归还到 train。")
else:
    print("当前 test 文件夹为空，无需归还。")


# =========================
# 第二步：重新随机抽取固定 test 集
# =========================

all_images = [f for f in os.listdir(src_img_dir) if f.lower().endswith(".jpg")]
all_images = sorted(all_images)

print("=" * 60)
print(f"当前 train 中总图片数: {len(all_images)}")

if len(all_images) < test_count:
    raise ValueError(f"当前图片数量 {len(all_images)} 小于 test_count={test_count}，无法抽取。")

# 检查标签是否完整
missing_labels = []

for img in all_images:
    lab = os.path.splitext(img)[0] + ".txt"
    lab_path = os.path.join(src_lab_dir, lab)

    if not os.path.exists(lab_path):
        missing_labels.append(lab)

if len(missing_labels) > 0:
    print("❌ 以下图片缺失标签：")
    for lab in missing_labels[:30]:
        print(lab)
    raise FileNotFoundError(f"共有 {len(missing_labels)} 个标签缺失，请先检查数据。")
else:
    print("✅ 所有图片均有对应标签。")

# 固定随机种子
random.seed(random_seed)

test_samples = random.sample(all_images, test_count)

print(f"准备抽取 test 集: {len(test_samples)} 张")

for img_name in test_samples:
    src_img_path = os.path.join(src_img_dir, img_name)
    dst_img_path = os.path.join(test_img_dir, img_name)

    lab_name = os.path.splitext(img_name)[0] + ".txt"
    src_lab_path = os.path.join(src_lab_dir, lab_name)
    dst_lab_path = os.path.join(test_lab_dir, lab_name)

    shutil.move(src_img_path, dst_img_path)
    shutil.move(src_lab_path, dst_lab_path)

print("=" * 60)
print(f"✅ 随机抽取完成！新的 test 集包含 {len(test_samples)} 张图片。")
print(f"当前 train 剩余图片数: {len([f for f in os.listdir(src_img_dir) if f.lower().endswith('.jpg')])}")
print(f"当前 test 图片数: {len([f for f in os.listdir(test_img_dir) if f.lower().endswith('.jpg')])}")
print("=" * 60)