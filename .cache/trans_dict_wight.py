from safetensors.torch import load_file
from safetensors.torch import save_file
import torch # 尽管代码里没直接用，但 safetensors 加载时需要它

# --- 关键代码开始 ---

# 1. 定义你的 .safetensors 文件路径
#    请确保将 "path/to/your_model.safetensors" 替换为你的实际文件路径
file_path = ".cache/bitnet-b1.58-2B-4T/model.safetensors"

# 2. 使用 load_file 函数加载文件
#    这个函数会直接返回一个字典，其中键是张量名，值是 torch.Tensor
model_dict = load_file(file_path)

new_model_dict = {}

print(f"成功加载文件: {file_path}")           

print(f"加载后对象的类型: {type(model_dict)}")

for key, value in model_dict.items():
    if key.endswith(".weight_scale"):
        continue
    if key.endswith(".weight"):
        scale_key = key.replace(".weight",".weight_scale")
        
        if scale_key in model_dict :
            new_model_dict[key] = model_dict[key].to( torch.bfloat16 )* model_dict[scale_key]
    else :
        new_model_dict[key] = model_dict[key].to( torch.bfloat16 )


output_file_path = ".cache/bitnet-b1.58-2B-4T/new_model.safetensors"

save_file(new_model_dict, output_file_path)

