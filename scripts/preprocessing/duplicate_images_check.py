import os
import hashlib
from collections import defaultdict
from pathlib import Path

# 替换为你存放 5500 张图片的真实文件夹路径
IMAGE_DIR = Path(r'F:\Program Files\Underwater_Detection\original_data\image') # 请修改此路径

def get_image_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def check_duplicates():
    print("🔍 正在全盘扫描 5500 张图片，寻找物理重复文件...")
    
    if not IMAGE_DIR.exists():
        print(f"❌ 找不到图片文件夹: {IMAGE_DIR}")
        return

    hash_dict = defaultdict(list)
    image_files = list(IMAGE_DIR.glob('*.jpg')) + list(IMAGE_DIR.glob('*.png'))
    
    for idx, img_path in enumerate(image_files):
        if idx % 500 == 0:
            print(f"  ...已扫描 {idx} 张")
        img_hash = get_image_hash(img_path)
        hash_dict[img_hash].append(img_path.name)

    duplicates = {k: v for k, v in hash_dict.items() if len(v) > 1}

    print("\n" + "="*50)
    if len(duplicates) > 0:
        print(f"🚨 警报！发现 {len(duplicates)} 组完全重复的图片！")
        for h, files in list(duplicates.items())[:5]: # 只打印前 5 组
            print(f"  - 重复组: {files}")
        print("  （这就解释了为什么模型结果会一模一样了）")
    else:
        print("✅ 没有发现‘物理层面’一模一样的图片。")
        print("💡 提示：如果没有物理重复，大概率是‘视频连续抽帧’导致的高度相似（请用肉眼幻灯片法验证）。")

if __name__ == '__main__':
    check_duplicates()