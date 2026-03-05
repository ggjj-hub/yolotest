from ultralytics import YOLO
import os

# 1. 加载你刚练好的"最强模型"
model = YOLO(r"D:\yolo11_test\runs\pose\classroom_action_v1\weights\best.pt")

# 2. 待标注的图片路径
source_img_dir = r"D:\yolo11_test\dataset_images"
save_txt_dir = r"D:\yolo11_test\final_project\auto_labels"

if not os.path.exists(save_txt_dir): os.makedirs(save_txt_dir)

# 3. 开始全自动"打工"
results = model.predict(
    source=source_img_dir,
    save_txt=True,       # 自动生成坐标 txt
    conf=0.4,            # 置信度阈值
    project=save_txt_dir,
    name="results"
)

print(f"全自动标注完成！结果保存在: {save_txt_dir}")
