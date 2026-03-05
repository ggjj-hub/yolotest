from ultralytics import YOLO

# 加载你练出来的"神种"模型
model = YOLO("runs/pose/classroom_action_v1/weights/best.pt")

# 预测一张新图
results = model.predict("D:/yolo11_test/some_new_frame.jpg", save=True)

# 查看结果：你会发现学生头顶的标签变成了具体的 action 名字
