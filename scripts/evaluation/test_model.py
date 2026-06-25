from ultralytics import YOLO
import pathlib

# 1. 设置路径
# 建议选你表现最好的那一折，比如第五折 (train_fold_5)
model_path = r'F:\Program Files\Underwater_Detection\runs\detect\train_fold_5\weights\best.pt'
test_data_path = r'F:\Program Files\Underwater_Detection\yolo_dataset\images\test'

# 2. 加载模型
model = YOLO(model_path)

# 3. 运行推理 (Predict)
results = model.predict(
    source=test_data_path,
    save=True,          # 保存可视化预测结果图
    save_txt=True,      # 保存检测到的坐标结果 (txt格式)
    conf=0.25,          # 置信度阈值，低于0.25的框会被过滤
    iou=0.45,           # NMS IoU阈值
    device=0,           # 使用 4060 显卡
    project='runs/test_results', # 结果保存的主目录
    name='fold_5_test'  # 本次测试的具体文件夹名
)

print(f"测试完成！结果已保存至 runs/test_results/fold_5_test")