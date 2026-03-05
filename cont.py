import os


def merge_and_generate_list(old_img_dir, new_img_dir, output_list_path):
    all_images = []

    # 扫描旧图片路径
    if os.path.exists(old_img_dir):
        for img in os.listdir(old_img_dir):
            if img.endswith(('.jpg', '.png', '.jpeg')):
                all_images.append(os.path.abspath(os.path.join(old_img_dir, img)))

    # 扫描新图片路径
    if os.path.exists(new_img_dir):
        for img in os.listdir(new_img_dir):
            if img.endswith(('.jpg', '.png', '.jpeg')):
                all_images.append(os.path.abspath(os.path.join(new_img_dir, img)))

    # 写入清单文件
    with open(output_list_path, 'w') as f:
        for item in all_images:
            f.write(item + '\n')

    print(f"✅ 合并完成！总计图片数量: {len(all_images)}")
    print(f"📝 清单文件已生成至: {output_list_path}")


# 修改为你实际的路径
merge_and_generate_list(
    old_img_dir=r"D:\yolo11_test\final_project\images",
    new_img_dir=r"D:\yolo11_test\new_data\images",
    output_list_path=r"D:\yolo11_test\train_v2.txt"
)