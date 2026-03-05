from ultralytics import YOLO

# 1. 加载模型
# yolo11n.pt 是轻量化版本，第一次运行会自动从官网下载
model = YOLO("yolo11n.pt")

# 2. 执行推理
# source: 你的视频文件名
# show=True: 实时弹出窗口显示检测画面
# save=True: 将检测后的视频保存到 runs/detect 目录下
# conf=0.25: 置信度阈值，低于 0.25 的目标会被过滤掉
results = model.predict(
    source="marseme_Sub_11.mp4",
    show=True,
    save=True,
    conf=0.25
)

print("处理完成！结果已保存至 runs 文件夹。")