from ultralytics import YOLO

def run_test():
    # 1. 加载模型
    model = YOLO(r'F:\Program Files\Underwater_Detection\runs\detect\train_fold_3\weights\best.pt')

    # 2. 执行评估
    metrics = model.val(
        data='test_only.yaml', 
        split='test',          
        device=0,
        imgsz=640,
        workers=2, # 限制读取线程，在你的电脑上更稳定
        save_json=True  # ✨ 加入这个参数，它会生成详细的 JSON 预测结果
    )

    # 3. 打印最终百分比
    print(f"\n测试集全类别 mAP50: {metrics.box.map50:.3f}")

# ✨ 关键：添加 Windows 必须的入口保护
if __name__ == '__main__':
    run_test()