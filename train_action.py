from ultralytics import YOLO

if __name__ == '__main__':
    # 加载之前训练得比较好的那个模型作为起点（预训练）
    model = YOLO(r"D:\yolo11_test\runs\pose\classroom_action_v1\weights\best.pt")

    results = model.train(
        data="final.yaml",
        epochs=150,
        imgsz=640,
        # --- 核心增强参数 ---
        scale=0.9,      # 允许图片缩放 90%，模拟近远变化
        mosaic=1.0,     # 强制开启 Mosaic，把大图切碎再拼起来学
        mixup=0.1,      # 增加重叠学习
        # ------------------
        batch=8,
        device=0,
        name="classroom_v2_final"
    )