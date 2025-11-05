from safetensors.torch import load_file, save_file

# 1. 定义文件路径（原文件路径和目标保存路径）
original_path = ".cache/ifairy-full-700M/model.safetensors"
# 同目录下的保存路径（可以是新文件名，也可以用原文件名覆盖）
save_path = ".cache/ifairy-full-700M/loaded_model.safetensors"  # 新文件名，避免覆盖原文件


# 2. 用load_file加载文件，得到权重字典
try:
    weights_dict = load_file(original_path)
    print(f"成功加载权重字典，包含 {len(weights_dict)} 个参数")
except Exception as e:
    print(f"加载失败：{e}")
    exit()  # 加载失败则退出


# 3. 把字典保存到同目录下（用save_file函数）
try:
    save_file(weights_dict, save_path)
    print(f"权重字典已保存到同目录：{save_path}")
except Exception as e:
    print(f"保存失败：{e}")