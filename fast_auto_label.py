from ultralytics import YOLO
import os

# 1. 配置路径
# 使用你刚才练出来的那个“懂你规矩”的模型
model_path = "D:\\yolo11_test\\runs\\pose\\my_custom_teacher\\weights\\best.pt"
source_images = "D:/yolo11_test/dataset_images/"
output_dir = "D:/yolo11_test/auto_dataset/"

if __name__ == '__main__':
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"错误：找不到模型 {model_path}，请确认训练已完成！")
    else:
        model = YOLO(model_path)

        # 2. 开启高速推理模式
        results = model.predict(
            source=source_images,
            save=False,          # 关键：不生成图片，提速 5 倍以上
            save_txt=True,       # 关键：只保存包含坐标的 txt 文件
            conf=0.3,            # 置信度阈值，过滤掉不确定的检测
            imgsz=640,           # 与训练分辨率保持一致
            device=0,            # 使用 GPU
            project=output_dir,  # 保存结果的总目录
            name="version_1"     # 具体的子目录名
        )

        print(f"全量标注完成！结果保存在: {output_dir}version_1/labels")
