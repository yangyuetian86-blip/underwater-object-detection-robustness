from ultralytics import YOLO
import multiprocessing

def train_model():
    # 加载模型
    model = YOLO("yolo11n.pt") 

    # 开始训练
    results = model.train(
        data="data.yaml", 
        epochs=100,        # 既然显卡已经就位，咱们跑 100 轮
        imgsz=640, 
        device=0,          # 确认使用 RTX 4060
        batch=16, 
        workers=4          # 多进程读取数据，必须配合下面的 __main__ 保护
    )

# 核心修复：必须把执行逻辑放在这个判断下面
if __name__ == '__main__':
    # 这一行是给 Windows 系统的额外保险
    multiprocessing.freeze_support()
    
    # 启动训练
    train_model()