import os


def check_yolo_pose_labels(label_dir, num_kpts=7):
    # YOLO Pose 格式标准长度: class(1) + box(4) + points(num_kpts * 3)
    expected_length = 1 + 4 + (num_kpts * 3)
    error_files = []

    for file in os.listdir(label_dir):
        if file.endswith('.txt'):
            path = os.path.join(label_dir, file)
            with open(path, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    data = line.strip().split()
                    if len(data) != expected_length:
                        print(f"!!! 格式错误: 文件 {file} 第 {i + 1} 行, 长度为 {len(data)} (预期 {expected_length})")
                        if file not in error_files:
                            error_files.append(file)

    if not error_files:
        print("✅ 所有标签格式正确！")
    else:
        print(f"\n❌ 共发现 {len(error_files)} 个异常文件，请立即修复。")


# 使用时修改为你的 labels 文件夹路径
check_yolo_pose_labels(r"D:\yolo11_test\final_project\labels")