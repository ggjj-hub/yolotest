from ultralytics import YOLO
import cv2

# 加载模型
model = YOLO('yolo11n-pose.pt')

# 设置窗口
WINDOW_NAME = "YOLOv11 Pose Tracking"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)  # 设置窗口大小

# 处理视频
results = model.track(
    source="D:\\yolo11_test\\115.mp4",
    show=False,  # 关闭内置显示
    save=True,
    stream=True,
    conf=0.5,
    iou=0.7,
    persist=True,
    tracker="bytetrack.yaml"
)

# yolo_pose_gpu_optimized.py
import torch
from ultralytics import YOLO
import cv2
import time
import numpy as np
import os

print("=" * 70)
print("YOLOv11 GPU加速姿态估计 - RTX 3050")
print("=" * 70)

# GPU信息显示
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ GPU设备: {torch.cuda.get_device_name(0)}")
print(f"✅ CUDA版本: {torch.version.cuda}")
print(f"✅ 计算能力: {torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}")
print(f"✅ GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

# 清空GPU缓存
torch.cuda.empty_cache()

# 设置优化选项
if device == 'cuda':
    torch.backends.cudnn.benchmark = True  # 启用cuDNN自动调优
    torch.backends.cuda.matmul.allow_tf32 = True  # 允许TF32加速
    print("✅ GPU优化已启用")

# 加载模型
print(f"\n🚀 加载YOLOv11姿态估计模型...")
model = YOLO('yolo11n-pose.pt').to(device)

# 如果模型文件不存在，会自动下载
if not os.path.exists('yolo11n-pose.pt'):
    print("正在下载yolo11n-pose.pt模型...")

# 设置视频路径
video_path = r"D:\yolo11_test\116.mp4"
if not os.path.exists(video_path):
    print(f"❌ 视频文件不存在: {video_path}")
    exit(1)

print(f"📹 处理视频: {video_path}")

# 创建窗口
WINDOW_NAME = f"YOLOv11 Pose - RTX 3050"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)

# GPU优化配置
tracking_config = {
    "source": video_path,
    "show": False,  # 关闭内置显示
    "save": True,  # 保存结果
    "stream": True,  # 流式处理
    "conf": 0.5,  # 置信度阈值
    "iou": 0.7,  # IOU阈值
    "persist": True,  # 保持追踪ID
    "tracker": "bytetrack.yaml",  # 使用ByteTrack追踪器
    "device": device,  # 使用GPU
    "half": True,  # 使用半精度(FP16)加速
    "imgsz": 640,  # 推理尺寸
    "max_det": 50,  # 最大检测数
    "agnostic_nms": True,  # 类别无关NMS
    "verbose": False,  # 减少输出
    "classes": None,  # 所有类别
    "save_txt": False,  # 不保存txt结果
    "save_conf": False,  # 不保存置信度
    "save_crop": False,  # 不保存裁剪
    "show_labels": True,  # 显示标签
    "show_conf": True,  # 显示置信度
    "show_boxes": True,  # 显示边界框
    "line_width": 2,  # 线宽
}

print("\n⚡ 开始GPU加速处理...")
print(f"   设备: {device.upper()}")
print(f"   半精度(FP16): 启用")
print(f"   推理尺寸: 640x640")
print(f"   追踪器: ByteTrack")

# 性能监控
frame_count = 0
fps_history = []
memory_history = []
start_time = time.time()
processing_start = time.time()

# 处理视频
results = model.track(**tracking_config)

# 逐帧处理
for result in results:
    frame_count += 1

    # 计算当前FPS
    current_time = time.time()
    elapsed = current_time - start_time
    current_fps = frame_count / elapsed if elapsed > 0 else 0
    fps_history.append(current_fps)

    # 获取GPU内存使用
    if device == 'cuda':
        memory_used = torch.cuda.memory_allocated() / 1024 ** 3
        memory_history.append(memory_used)

    # 获取标注的帧
    frame = result.plot()

    # 添加信息叠加层
    overlay = frame.copy()

    # 信息背景
    cv2.rectangle(overlay, (5, 5), (350, 140), (0, 0, 0, 180), -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    # 显示信息
    info_lines = [
        f"设备: RTX 3050 (CUDA {torch.version.cuda})",
        f"帧: {frame_count}",
        f"实时FPS: {current_fps:.1f}",
    ]

    if device == 'cuda':
        info_lines.append(f"GPU内存: {memory_used:.2f} GB")

    if result.boxes is not None and result.boxes.id is not None:
        track_ids = result.boxes.id.int().cpu().tolist()
        info_lines.append(f"追踪人数: {len(track_ids)}")

        # 显示前5个追踪ID
        if len(track_ids) > 0:
            ids_str = ", ".join(map(str, track_ids[:5]))
            if len(track_ids) > 5:
                ids_str += f" ... (+{len(track_ids) - 5})"
            info_lines.append(f"追踪ID: {ids_str}")

    # 绘制信息文本
    for i, text in enumerate(info_lines):
        y_pos = 30 + i * 20
        cv2.putText(frame, text, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 显示进度条
    if hasattr(result, 'speed'):
        infer_time = result.speed.get('inference', 0)
        if infer_time > 0:
            progress_width = 300
            progress_height = 10
            progress_x = 10
            progress_y = 150

            # 绘制进度条背景
            cv2.rectangle(frame, (progress_x, progress_y),
                          (progress_x + progress_width, progress_y + progress_height),
                          (100, 100, 100), -1)

            # 计算进度（基于推理时间，假设目标FPS为30）
            progress = min(1.0, 30 / (1000 / infer_time) if infer_time > 0 else 0)
            fill_width = int(progress_width * progress)

            # 绘制进度条填充
            color = (0, 255, 0) if progress > 0.7 else (0, 255, 255) if progress > 0.4 else (0, 165, 255)
            cv2.rectangle(frame, (progress_x, progress_y),
                          (progress_x + fill_width, progress_y + progress_height),
                          color, -1)

            # 添加进度文本
            progress_text = f"推理: {infer_time:.1f}ms ({progress * 100:.0f}%)"
            cv2.putText(frame, progress_text, (progress_x, progress_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 显示帧
    cv2.imshow(WINDOW_NAME, frame)

    # 按键控制
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 退出
        break
    elif key == ord('f'):  # 切换全屏
        is_fullscreen = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
        if is_fullscreen == cv2.WINDOW_NORMAL:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    elif key == ord('p'):  # 暂停
        cv2.waitKey(0)
    elif key == ord('s'):  # 保存截图
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"frame_{frame_count:06d}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 已保存截图: {filename}")
    elif key == ord('r'):  # 重置窗口
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    elif key == ord('1'):  # 窗口大小预设1
        cv2.resizeWindow(WINDOW_NAME, 960, 540)
    elif key == ord('2'):  # 窗口大小预设2
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    elif key == ord('3'):  # 窗口大小预设3
        cv2.resizeWindow(WINDOW_NAME, 1600, 900)

    # 每30帧输出一次进度
    if frame_count % 30 == 0:
        avg_fps = np.mean(fps_history[-30:]) if len(fps_history) >= 30 else current_fps
        if device == 'cuda':
            avg_memory = np.mean(memory_history[-30:]) if memory_history else 0
            print(f"📊 进度: {frame_count}帧, FPS: {avg_fps:.1f}, GPU内存: {avg_memory:.2f}GB")
        else:
            print(f"📊 进度: {frame_count}帧, FPS: {avg_fps:.1f}")

# 清理
cv2.destroyAllWindows()

# 计算总时间
total_time = time.time() - processing_start

# 性能报告
if fps_history:
    avg_fps = np.mean(fps_history)
    min_fps = np.min(fps_history)
    max_fps = np.max(fps_history)

    print("\n" + "=" * 70)
    print("🎯 性能报告")
    print("=" * 70)
    print(f"📈 总帧数: {frame_count}")
    print(f"⏱️  总时间: {total_time:.2f}秒")
    print(f"🚀 平均FPS: {avg_fps:.1f}")
    print(f"📉 最低FPS: {min_fps:.1f}")
    print(f"📈 最高FPS: {max_fps:.1f}")
    print(f"🎯 处理速度: {frame_count / total_time:.1f} FPS (平均)")

    if device == 'cuda':
        max_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
        avg_memory = np.mean(memory_history) if memory_history else 0
        print(f"💾 最大GPU内存: {max_memory:.2f} GB")
        print(f"💾 平均GPU内存: {avg_memory:.2f} GB")

        # 计算GPU利用率
        if total_time > 0:
            gpu_utilization = (avg_fps / 60) * 100  # 假设60FPS为100%利用率
            print(f"⚡ GPU利用率: {min(gpu_utilization, 100):.1f}%")

    print(f"✅ 处理完成!")
    print("=" * 70)

    # 保存性能报告
    report_filename = f"performance_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_filename, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("YOLOv11 GPU性能报告\n")
        f.write("=" * 70 + "\n")
        f.write(f"GPU设备: {torch.cuda.get_device_name(0)}\n")
        f.write(f"总帧数: {frame_count}\n")
        f.write(f"总时间: {total_time:.2f}秒\n")
        f.write(f"平均FPS: {avg_fps:.1f}\n")
        f.write(f"最低FPS: {min_fps:.1f}\n")
        f.write(f"最高FPS: {max_fps:.1f}\n")
        if device == 'cuda':
            f.write(f"最大GPU内存: {max_memory:.2f} GB\n")
            f.write(f"平均GPU内存: {avg_memory:.2f} GB\n")
        f.write("=" * 70 + "\n")

    print(f"📄 性能报告已保存: {report_filename}")

# 清空GPU缓存
if device == 'cuda':
    torch.cuda.empty_cache()
    print("🧹 GPU缓存已清理")

print("\n✨ 所有任务完成!")