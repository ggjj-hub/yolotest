import cv2
import os
import sys


def extract_frames(video_path, output_folder, interval_seconds=2):
    """
    video_path: 视频文件路径
    output_folder: 保存图片的文件夹路径
    interval_seconds: 每隔几秒截图一张
    """

    # 1. 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"❌ 错误：找不到视频文件！请检查路径是否正确：")
        print(f"   当前路径: {video_path}")
        return

    # 2. 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 已创建输出文件夹: {output_folder}")

    # 3. 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 错误：OpenCV 无法打开视频，请检查视频格式或是否被占用。")
        return

    # 获取视频基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
    duration = total_frames / fps  # 总时长（秒）
    frame_interval = int(fps * interval_seconds)  # 抽帧步长

    print(f"✅ 视频打开成功！")
    print(f"📊 视频信息: 时长 {duration / 60:.2f} 分钟 | 帧率 {fps:.2f} | 总帧数 {total_frames}")
    print(f"📸 抽帧设置: 每 {interval_seconds} 秒一张，预计提取 {int(duration / interval_seconds)} 张图")
    print("-" * 50)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 4. 按间隔保存图片
        if frame_count % frame_interval == 0:
            # 命名格式：frame_0001.jpg
            image_name = f"frame_{saved_count:04d}.jpg"
            save_path = os.path.join(output_folder, image_name)

            # 保存图片
            cv2.imwrite(save_path, frame)
            saved_count += 1

            # 在控制台实时显示进度
            sys.stdout.write(f"\r正在提取第 {saved_count} 张图片... 进度: {frame_count / total_frames * 100:.1f}%")
            sys.stdout.flush()

        frame_count += 1

    cap.release()
    print(f"\n\n✨ 任务完成！")
    print(f"✅ 共提取 {saved_count} 张图片。")
    print(f"📂 图片保存位置: {os.path.abspath(output_folder)}")


# ==================== 配置区域 ====================
if __name__ == "__main__":
    # 使用 r 前缀处理带空格的 Windows 路径，防止转义错误
    input_video = r"D:\data source\students-full.mp4"

    # 建议保存到你项目下的目录
    output_dir = r"D:\yolo11_test\dataset_images"

    # 执行抽帧（建议2秒一张，如果想要更多数据可改为1）
    extract_frames(input_video, output_dir, interval_seconds=2)