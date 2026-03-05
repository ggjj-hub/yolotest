from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载官方预训练模型作为底座
    model = YOLO("yolo11n-pose.pt")

    # 2. 训练（微调）
    model.train(
        data="seed.yaml",
        epochs=100,      # 数据量少，100轮能让模型记住你的连接逻辑
        imgsz=640,
        device=0,        # 使用你的 3050 显卡
        batch=4,         # 显存够用，4比较稳
        name="my_custom_teacher"
    )