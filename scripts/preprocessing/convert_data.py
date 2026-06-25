import os
import xml.etree.ElementTree as ET
import shutil
from PIL import Image  # 需要用到这个库来读取图片真实尺寸

# --- 配置部分 ---
class_mapping = {
    'holothurian': 0,
    'echinus': 1,
    'scallop': 2,
    'starfish': 3
}

# 路径设置
xml_dir = './original_data/box/'
img_dir = './original_data/image/'
yolo_dataset_dir = './yolo_dataset/'
yolo_images_dir = os.path.join(yolo_dataset_dir, 'images/train/')
yolo_labels_dir = os.path.join(yolo_dataset_dir, 'labels/train/')

os.makedirs(yolo_images_dir, exist_ok=True)
os.makedirs(yolo_labels_dir, exist_ok=True)

def convert_coordinates(size, box):
    """
    size: (真实宽, 真实高)
    box: (xmin, ymin, xmax, ymax)
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    
    # 强制限制在 0-1 之间，防止微小误差导致的越界
    return (min(0.999999, max(0, x * dw)), 
            min(0.999999, max(0, y * dh)), 
            min(0.999999, max(0, w * dw)), 
            min(0.999999, max(0, h * dh)))

def run_conversion():
    valid_count = 0
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    print(f"开始处理，共发现 {len(xml_files)} 个 XML 文件...")

    for xml_filename in xml_files:
        xml_path = os.path.join(xml_dir, xml_filename)
        img_filename = xml_filename.replace('.xml', '.jpg')
        src_img_path = os.path.join(img_dir, img_filename)

        if not os.path.exists(src_img_path):
            continue

        # --- 核心改进：实时读取图片真实尺寸 ---
        try:
            with Image.open(src_img_path) as img:
                real_width, real_height = img.size
        except Exception as e:
            print(f"无法读取图片 {img_filename}: {e}")
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()
        yolo_labels = []

        for obj in root.findall('object'):
            name = obj.find('name').text.strip().lower()
            if name == 'waterweeds' or name not in class_mapping:
                continue

            class_id = class_mapping[name]
            xmlbox = obj.find('bndbox')
            
            # 提取坐标
            b = (float(xmlbox.find('xmin').text),
                 float(xmlbox.find('ymin').text),
                 float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymax').text))

            # 使用读取到的 real_width, real_height 转换
            bb = convert_coordinates((real_width, real_height), b)
            yolo_labels.append(f"{class_id} " + " ".join([f"{a:.6f}" for a in bb]))

        if yolo_labels:
            # 拷贝图片并写入标签
            dst_img = os.path.join(yolo_images_dir, img_filename)
            shutil.copy(src_img_path, dst_img)
            
            txt_filename = xml_filename.replace('.xml', '.txt')
            with open(os.path.join(yolo_labels_dir, txt_filename), 'w') as f:
                f.write('\n'.join(yolo_labels))
            valid_count += 1

    print(f"--- 处理完成 ---")
    print(f"成功导入有效数据: {valid_count} 组")

if __name__ == "__main__":
    run_conversion()