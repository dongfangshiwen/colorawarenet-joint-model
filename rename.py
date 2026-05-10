import os

def rename_images(folder_path):
    # 支持的图片扩展名
    exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    # 获取所有图片文件并排序
    files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in exts]
    files.sort()

    # 逐个重命名
    for i, filename in enumerate(files, start=1):
        ext = os.path.splitext(filename)[1].lower()
        new_name = f"{i:03d}.png"  # 格式化为 001.png, 002.png ...
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"{filename} -> {new_name}")

    print("重命名完成！")

if __name__ == "__main__":
    # 修改为你的文件夹路径，例如：
    folder = r"datasets\clear"
    rename_images(folder)
